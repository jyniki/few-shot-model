import torch
import torch.nn as nn
import torch.nn.functional as F
import models
import utils
from .models import register

@register('meta-co')
class MetaCo(nn.Module):
    def __init__(self, encoder, encoder_args={}, method='cos', temp=10., temp_learnable=True):
        super().__init__()
        self.encoder_q = models.make(encoder, **encoder_args)  # Query Encoder
        self.encoder_k = models.make(encoder, **encoder_args)  # Key Encoder
        self.method = method
        self.K = 10000   # number of negative keys (default: 1000)
        self.m = 0.999   # moco momentum of updating key encoder (default: 0.999)
        self.T = 0.07    # softmax temperature (default: 0.07)

        # Initialize the key encoder to have the same values as query encoder
        # Do not update the key encoder via gradient
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False     # not update by gradient

        # create the queue 
        self.register_buffer("queue", torch.randn(self.encoder_q.out_dim, self.K)) 
        self.queue = nn.functional.normalize(self.queue, dim=0)

        # Create pointer to store current position in the queue when enqueue and dequeue
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        if temp_learnable:
            self.temp = nn.Parameter(torch.tensor(temp))
        else:
            self.temp = temp

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        """

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(im_k)            # keys: NxC
            k = nn.functional.normalize(k, dim=1)
        
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # # dequeue and enqueue
        # self._dequeue_and_enqueue(k)

        return logits, labels


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output
