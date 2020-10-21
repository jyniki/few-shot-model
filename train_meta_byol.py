import argparse
import os
import yaml

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from tensorboardX import SummaryWriter

import models
import utils
from fewshot_dataloader import get_fewshot_dataset, get_meta_loader, split_shot_query, make_nk_label

def main(config):
    svname = args.name
    if svname is None:
        svname = 'meta_{}-{}shot'.format(
            config['train_dataset'], config['n_shot'])
        svname += '_' + config['model'] + '-' + config['model_args']['encoder']
    if args.tag is not None:
        svname += '_' + args.tag
    save_path = os.path.join('./save', svname)
    utils.set_save_path(save_path)
    utils.set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    yaml.dump(config, open(os.path.join(save_path, 'meta_config.yaml'), 'w'))

    #### Dataset ####
    n_way, n_shot = config['n_way'], config['n_shot']
    n_query = config['n_query']

    #### Make datasets ####
    # train
    train_dataset = get_fewshot_dataset(config['dataset_path'], config['train_dataset'], **config['train_dataset_args'])
    train_dataloader = get_meta_loader(train_dataset, n_batch=200, ways=n_way, shots=n_shot, query_shots=n_query, batch_size=config['train_dataset_args']['batch_size'],num_workers=4)
    utils.log('train dataset: {}'.format(config['train_dataset']))
    if config.get('visualize_datasets'):
        utils.visualize_dataset(train_dataset, 'train_dataset', writer)

    #### Model and optimizer ####
    if config.get('load'):
        model_sv = torch.load(config['load'])
        model = models.load(model_sv)
    else:
        model = models.make(config['model'], **config['model_args'])

        if config.get('load_encoder'):
            encoder = models.load(torch.load(config['load_encoder'])).encoder
            model.online_encoder.load_state_dict(encoder.state_dict())

    if config.get('_parallel'):
        model = nn.DataParallel(model)

    utils.log('num params: {}'.format(utils.compute_n_params(model)))

    optimizer, lr_scheduler = utils.make_optimizer(
            model.parameters(),
            config['optimizer'], **config['optimizer_args'])

    #### train and test ####
    max_epoch = config['max_epoch']
    save_epoch = config.get('save_epoch')
    if save_epoch is None:
        save_epoch = 5
    max_va = 0.
    timer_used = utils.Timer()
    timer_epoch = utils.Timer()

    # recode loss and accuracy
    aves_keys = ['tl', 'ta', 'vl', 'va', 'tvl', 'tva']
    trlog = dict()
    for k in aves_keys:
        trlog[k] = []

    for epoch in range(1, max_epoch + 1):
        timer_epoch.s()
        aves = {k: utils.Averager() for k in aves_keys}

        # train
        model.train()
        if config.get('freeze_bn'):
            utils.freeze_bn(model) 
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        np.random.seed(epoch)

        '''
        for data, _ in tqdm(train_dataloader, desc='train', leave=False):
            data[0] = data[0].cuda()
            data[1] = data[1].cuda()

            loss = model(data[0], data[1]).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            aves['tl'].add(loss.item())
        
        '''
        for data, _ in tqdm(train_dataloader, desc='train', leave=False):
            data[0] = data[0].cuda()
            data[1] = data[1].cuda()
            x_shot_one, x_query_one = split_shot_query(data[0], n_way, n_shot, n_query, ep_per_batch=config['train_dataset_args']['batch_size'])
            x_shot_two, x_query_two = split_shot_query(data[1], n_way, n_shot, n_query, ep_per_batch=config['train_dataset_args']['batch_size'])
            label = make_nk_label(n_way, 2 * n_query, ep_per_batch=config['train_dataset_args']['batch_size']).cuda()
            loss_byol, logits = model(x_shot_one, x_query_one, x_shot_two, x_query_two)

            loss_meta = F.cross_entropy(logits.view(-1, n_way), label)
            loss = loss_byol.mean() + loss_meta
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.module.update_moving_average()
            aves['tl'].add(loss.item())
       

        _sig = int(_[-1])

        # post
        if lr_scheduler is not None:
            lr_scheduler.step()

        for k, v in aves.items():
            aves[k] = v.item()
            trlog[k].append(aves[k])

        t_epoch = utils.time_str(timer_epoch.t())
        t_used = utils.time_str(timer_used.t())
        t_estimate = utils.time_str(timer_used.t() / epoch * max_epoch)
       
        utils.log('epoch {}, train {:.4f}|{:.4f}, {} {}/{} (@{})'.format(epoch, aves['tl'], aves['ta'], t_epoch, t_used, t_estimate, _sig))

        writer.add_scalars('loss', {'train': aves['tl'],}, epoch)
        writer.add_scalars('acc', {'train': aves['ta'],}, epoch)

        if config.get('_parallel'):
            model_ = model.module
        else:
            model_ = model

        training = {
            'epoch': epoch,
            'optimizer': config['optimizer'],
            'optimizer_args': config['optimizer_args'],
            'optimizer_sd': optimizer.state_dict(),
        }
        save_obj = {
            'file': __file__,
            'config': config,

            'model': config['model'],
            'model_args': config['model_args'],
            'model_sd': model_.state_dict(),

            'training': training,
        }
        torch.save(save_obj, os.path.join(save_path, 'epoch-last.pth'))
        torch.save(trlog, os.path.join(save_path, 'trlog.pth'))

        if (save_epoch is not None) and epoch % save_epoch == 0:
            torch.save(save_obj,
                    os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))

        if aves['va'] > max_va:
            max_va = aves['va']
            torch.save(save_obj, os.path.join(save_path, 'max-va.pth'))

        writer.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/train_meta_byol_mini.yaml')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0,1,2,3')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    if len(args.gpu.split(',')) > 1:
        config['_parallel'] = True
        config['_gpu'] = args.gpu

    utils.set_gpu(args.gpu)
    main(config)

