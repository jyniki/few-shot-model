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


@register('meta-byol2')
class MetaByol5(nn.Module):

    def __init__(self, encoder, encoder_args={}, method='cos',
                 temp=10., temp_learnable=True):
        super().__init__()
        self.method = method
        if temp_learnable:
            self.temp = nn.Parameter(torch.tensor(temp))
        else:
            self.temp = temp
        self.online_encoder = models.make(encoder, **encoder_args)
        self.online_projector = MLP(self.online_encoder.out_dim, 256, 4096).cuda()
        self.online_encoder_all = nn.Sequential(self.online_encoder, self.online_projector)

        self.target_encoder_all = None

        self.target_ema_updater = EMA(0.99)

        self.online_predictor = MLP(256, 256, 4096).cuda()

        # send a mock image tensor to instantiate singleton parameters
        self.forward(torch.randn(1, 5, 1, 3, 84, 84).cuda(), torch.randn(1, 75, 3, 84, 84).cuda(),
                     torch.randn(1, 5, 1, 3, 84, 84).cuda(), torch.randn(1, 75, 3, 84, 84).cuda(),
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

    def forward(self, x_shot_ori, x_query_ori, x_shot_one, x_query_one, x_shot_two, x_query_two):

        img_shape = x_shot_one.shape[-3:]
        x_shot_one = x_shot_one.view(-1, *img_shape)
        x_query_one = x_query_one.view(-1, *img_shape)
        x_shot_two = x_shot_two.view(-1, *img_shape)
        x_query_two = x_query_two.view(-1, *img_shape)

        image_one = torch.cat([x_shot_one, x_query_one], dim=0)
        image_two = torch.cat([x_shot_two, x_query_two], dim=0)

        online_feat_one = self.online_encoder(image_one)
        online_feat_two = self.online_encoder(image_two)

        online_proj_one = self.online_projector(online_feat_one)
        online_proj_two = self.online_projector(online_feat_two)

        online_pred_one = self.online_predictor(online_proj_one)
        online_pred_two = self.online_predictor(online_proj_two)

        with torch.no_grad():
            target_encoder_all = self._get_target_encoder()
            target_proj_one = target_encoder_all(image_one)
            target_proj_two = target_encoder_all(image_two)

        loss_one = loss_fn(online_pred_one, target_proj_two.detach())
        loss_two = loss_fn(online_pred_two, target_proj_one.detach())

        loss_byol = loss_one + loss_two

        shot_shape = x_shot_ori.shape[:-3]
        query_shape = x_query_ori.shape[:-3]
        img_shape = x_shot_ori.shape[-3:]

        x_shot = x_shot_ori.view(-1, *img_shape)
        x_query = x_query_ori.view(-1, *img_shape)
        x_tot = self.online_encoder(torch.cat([x_shot, x_query], dim=0))
        x_shot, x_query = x_tot[:len(x_shot)], x_tot[-len(x_query):]
        x_shot = x_shot.view(*shot_shape, -1)
        x_query = x_query.view(*query_shape, -1)

        if self.method == 'cos':
            x_shot = x_shot.mean(dim=-2)
            x_shot = F.normalize(x_shot, dim=-1)
            x_query = F.normalize(x_query, dim=-1)
            metric = 'dot'
        elif self.method == 'sqr':
            x_shot = x_shot.mean(dim=-2)
            metric = 'sqr'

        logits = utils.compute_logits(
            x_query, x_shot, metric=metric, temp=self.temp)

        return loss_byol, logits

    def test(self, x_shot, x_query):
        shot_shape = x_shot.shape[:-3]
        query_shape = x_query.shape[:-3]
        img_shape = x_shot.shape[-3:]

        x_shot = x_shot.view(-1, *img_shape)
        x_query = x_query.view(-1, *img_shape)
        x_tot = self.online_encoder(torch.cat([x_shot, x_query], dim=0))
        x_shot, x_query = x_tot[:len(x_shot)], x_tot[-len(x_query):]
        x_shot = x_shot.view(*shot_shape, -1)
        x_query = x_query.view(*query_shape, -1)

        if self.method == 'cos':
            x_shot = x_shot.mean(dim=-2)
            x_shot = F.normalize(x_shot, dim=-1)
            x_query = F.normalize(x_query, dim=-1)
            metric = 'dot'
        elif self.method == 'sqr':
            x_shot = x_shot.mean(dim=-2)
            metric = 'sqr'

        logits = utils.compute_logits(
            x_query, x_shot, metric=metric, temp=self.temp)
        return logits


def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)