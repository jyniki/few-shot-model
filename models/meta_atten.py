import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import models
import utils
from .models import register

def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(dim=1).view(-1, 1)
    return distance_matrix

@register('meta_atten')
class MetaAttention(nn.Module):
    def __init__(self, encoder, encoder_args={}, method='cos', temp=10., temp_learnable=True):
        super().__init__()
        self.encoder = models.make(encoder, **encoder_args)  
        self.method = method
        if temp_learnable:
            self.temp = nn.Parameter(torch.tensor(temp))
        else:
            self.temp = temp

        self.avg = nn.AvgPool2d(kernel_size=14, stride=1)
        self.map1 = nn.Linear(2048 * 2, 512)
        self.map2 = nn.Linear(512, 2048)
        self.fc = nn.Linear(2048, 200)
        self.drop = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_shot, x_query):
        shot_shape = x_shot.shape[:-3]
        query_shape = x_query.shape[:-3]
        img_shape = x_shot.shape[-3:]
        x_shot = x_shot.view(-1, *img_shape)    # [5,3,84,84]
        x_query = x_query.view(-1, *img_shape)  # [75,3,84,84] 
        x_tot = self.encoder(torch.cat([x_shot, x_query], dim=0))  # [80,512]
        x_shot, x_query = x_tot[:len(x_shot)], x_tot[-len(x_query):]
        x_shot = x_shot.view(*shot_shape, -1)   # [1,5,1,512]
        x_query = x_query.view(*query_shape, -1)  # [1,75,512]

        conv_out = self.encoder(images)
        pool_out = self.avg(conv_out).squeeze()
        intra_pairs, inter_pairs, intra_labels, inter_labels = self.get_pairs(pool_out, targets)
        features1 = torch.cat([pool_out[intra_pairs[:, 0]], pool_out[inter_pairs[:, 0]]], dim=0)
        features2 = torch.cat([pool_out[intra_pairs[:, 1]], pool_out[inter_pairs[:, 1]]], dim=0)
        labels1 = torch.cat([intra_labels[:, 0], inter_labels[:, 0]], dim=0)
        labels2 = torch.cat([intra_labels[:, 1], inter_labels[:, 1]], dim=0)
        mutual_features = torch.cat([features1, features2], dim=1)
        map1_out = self.map1(mutual_features)
        map2_out = self.drop(map1_out)
        map2_out = self.map2(map2_out)
        gate1 = torch.mul(map2_out, features1)
        gate1 = self.sigmoid(gate1)
        gate2 = torch.mul(map2_out, features2)
        gate2 = self.sigmoid(gate2)
        features1_self = torch.mul(gate1, features1) + features1
        features1_other = torch.mul(gate2, features1) + features1
        features2_self = torch.mul(gate2, features2) + features2
        features2_other = torch.mul(gate1, features2) + features2
        logit1_self = self.fc(self.drop(features1_self))
        logit1_other = self.fc(self.drop(features1_other))
        logit2_self = self.fc(self.drop(features2_self))
        logit2_other = self.fc(self.drop(features2_other))
        
        if self.method == 'cos':
            x_shot = x_shot.mean(dim=-2)            # [1,5,512]
            x_shot = F.normalize(x_shot, dim=-1)    # [1,5,512]
            x_query = F.normalize(x_query, dim=-1)  # [1,75,512]
            metric = 'dot'
            
        elif self.method == 'sqr':
            x_shot = x_shot.mean(dim=-2)
            metric = 'sqr'

        return logit1_self, logit1_other, logit2_self, logit2_other, labels1, labels2

    def get_pairs(self, embeddings, labels):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        distance_matrix = pdist(embeddings).detach().cpu().numpy()
        labels = labels.detach().cpu().numpy().reshape(-1,1)
        num = labels.shape[0]
        dia_inds = np.diag_indices(num)
        lb_eqs = (labels == labels.T)
        lb_eqs[dia_inds] = False
        dist_same = distance_matrix.copy()
        dist_same[lb_eqs == False] = np.inf
        intra_idxs = np.argmin(dist_same, axis=1)
        dist_diff = distance_matrix.copy()
        lb_eqs[dia_inds] = True
        dist_diff[lb_eqs == True] = np.inf
        inter_idxs = np.argmin(dist_diff, axis=1)
        intra_pairs = np.zeros([embeddings.shape[0], 2])
        inter_pairs  = np.zeros([embeddings.shape[0], 2])
        intra_labels = np.zeros([embeddings.shape[0], 2])
        inter_labels = np.zeros([embeddings.shape[0], 2])
        for i in range(embeddings.shape[0]):
            intra_labels[i, 0] = labels[i]
            intra_labels[i, 1] = labels[intra_idxs[i]]
            intra_pairs[i, 0] = i
            intra_pairs[i, 1] = intra_idxs[i]
            inter_labels[i, 0] = labels[i]
            inter_labels[i, 1] = labels[inter_idxs[i]]
            inter_pairs[i, 0] = i
            inter_pairs[i, 1] = inter_idxs[i]
        intra_labels = torch.from_numpy(intra_labels).long().to(device)
        intra_pairs = torch.from_numpy(intra_pairs).long().to(device)
        inter_labels = torch.from_numpy(inter_labels).long().to(device)
        inter_pairs = torch.from_numpy(inter_pairs).long().to(device)
        return intra_pairs, inter_pairs, intra_labels, inter_labels



















