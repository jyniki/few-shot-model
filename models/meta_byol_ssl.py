import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from .models import register
import copy
import random
from functools import wraps

import utils

# exponential moving average
class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model):
    for i, (current_params, ma_params) in enumerate(zip(current_model.parameters(), ma_model.parameters())):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance
        return wrapper
    return inner_fn

class MLP(nn.Module):
    def __init__(self, dim, projection_size, hidden_size = 4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)

@register('meta-byol-ssl')
class MetaByol(nn.Module):
    def __init__(self, encoder, encoder_args={}):
        super().__init__()
        self.method = method
        moving_average_decay = 0.99
        projection_size = 256
        projection_hidden_size = 4096
        if temp_learnable:
            self.temp = nn.Parameter(torch.tensor(temp))
        else:
            self.temp = temp
        self.online_encoder = models.make(encoder, **encoder_args)
        self.online_projector = MLP(self.online_encoder.out_dim, projection_size, projection_hidden_size).cuda()
        self.online_encoder_all = nn.Sequential(self.online_encoder, self.online_projector)

        self.target_encoder_all = None
        self.target_ema_updater = EMA(moving_average_decay)
        
        self.online_predictor = MLP(projection_size, projection_size, projection_hidden_size).cuda()

        # send a mock image tensor to instantiate singleton parameters
        # x_shot_one, x_query_one, x_shot_two, x_query_two
        self.forward(torch.randn(1, 5, 1, 3, 84, 84).cuda(), torch.randn(1, 75, 3, 84, 84).cuda(),
                     torch.randn(1, 5, 1, 3, 84, 84).cuda(), torch.randn(1, 75, 3, 84, 84).cuda())

    @singleton('target_encoder_all')
    def _get_target_encoder(self):
        target_encoder_all = copy.deepcopy(self.online_encoder_all)
        return target_encoder_all

    def reset_moving_average(self):
        del self.target_encoder_all
        self.target_encoder_all = None

    def update_moving_average(self):
        assert self.target_encoder_all is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder_all, self.online_encoder_all)

    
    def forward(self, x_shot_one, x_query_one, x_shot_two, x_query_two):
        
        shot_shape = x_shot_one.shape[:-3]
        query_shape = x_query_one.shape[:-3]
        img_shape = x_shot_one.shape[-3:]

        x_shot_one = x_shot_one.view(-1, *img_shape)   # [5,3,84,84]
        x_query_one = x_query_one.view(-1, *img_shape) # [75,3,84,84]
        x_shot_two = x_shot_one.view(-1, *img_shape)
        x_query_two = x_query_two.view(-1, *img_shape)

        # contrastive learning in query + support
        image_one = torch.cat([x_shot_one, x_query_one], dim=0) # [80,3,84,84]
        image_two = torch.cat([x_shot_two, x_query_two], dim=0)
        
        online_feat_one = self.online_encoder(image_one)   # [80,512]
        online_feat_two = self.online_encoder(image_two)
        
        online_proj_one = self.online_projector(online_feat_one) # [80,256]
        online_proj_two = self.online_projector(online_feat_two)
        
        online_pred_one = self.online_predictor(online_proj_one) # [80,256]
        online_pred_two = self.online_predictor(online_proj_two)

        with torch.no_grad():
            target_encoder_all = self._get_target_encoder()
            target_proj_one = target_encoder_all(image_one)  # [80,256]
            target_proj_two = target_encoder_all(image_two)

        loss_one = loss_fn(online_pred_one, target_proj_two.detach())
        loss_two = loss_fn(online_pred_two, target_proj_one.detach())
        
        loss_byol = loss_one + loss_two
       
        return loss_byol
  
def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

