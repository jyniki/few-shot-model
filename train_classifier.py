import argparse
import os
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter

import models
import utils
from fewshot_dataloader import get_fewshot_dataset, get_meta_loader, get_standard_loader, split_shot_query, make_nk_label

def main(config):
    svname = args.name
    if svname is None:
        svname = 'classifier_{}'.format(config['train_dataset'])
        svname += '_' + config['model_args']['encoder']
        clsfr = config['model_args']['classifier']
        if clsfr != 'linear-classifier':
            svname += '-' + clsfr
    if args.tag is not None:
        svname += '_' + args.tag
    save_path = os.path.join('./save', svname)
    utils.set_save_path(save_path)
    utils.set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    yaml.dump(config, open(os.path.join(save_path, 'config.yaml'), 'w'))

    #### Dataset ####

    # train  
    train_dataset = get_fewshot_dataset(config['dataset_path'],config['train_dataset'], **config['train_dataset_args'])
    train_loader = get_standard_loader(train_dataset, config['train_dataset_args']['batch_size'], shuffle=True, num_workers=8)
    utils.log('train dataset: {} (x{}), {}'.format(train_dataset[0][0].shape, len(train_dataset), train_dataset.n_classes))

    if config.get('visualize_datasets'):
        utils.visualize_dataset(train_dataset, 'train_dataset', writer)  # visualize in tensorboard

    # val 
    if config.get('val_dataset'):
        eval_val = True
        val_dataset = get_fewshot_dataset(config['dataset_path'], config['val_dataset'], **config['val_dataset_args'])
        val_loader = get_standard_loader(val_dataset, config['val_dataset_args']['batch_size'], num_workers=8)
        utils.log('val dataset: {} (x{}), {}'.format(val_dataset[0][0].shape, len(val_dataset), val_dataset.n_classes))

        if config.get('visualize_datasets'):
            utils.visualize_dataset(val_dataset, 'val_dataset', writer)
    else:
        eval_val = False

    # few-shot eval
    if config.get('fs_dataset'):
        ef_epoch = config.get('eval_fs_epoch')
        if ef_epoch is None:
            ef_epoch = 5
        eval_fs = True
        fs_dataset = get_fewshot_dataset(config['dataset_path'], config['fs_dataset'], **config['fs_dataset_args'])
        utils.log('fs dataset: {} (x{}), {}'.format(fs_dataset[0][0].shape, len(fs_dataset),fs_dataset.n_classes))

        if config.get('visualize_datasets'):
            utils.visualize_dataset(fs_dataset, 'fs_dataset', writer)

        n_way = 5  # random choose 5 classes
        n_query = 15
        n_shots = [1, 5]  # 1-shot or 5-shot
        fs_loaders = []
        for n_shot in n_shots:
            fs_loader = get_meta_loader(fs_dataset, n_batch=200, ways=n_way, shots=n_shot, query_shots=n_query, batch_size=config['ep_per_batch'], num_workers=8)
            fs_loaders.append(fs_loader)
    else:
        eval_fs = False


    #### Model and Optimizer ####
    if config.get('load'):
        model_sv = torch.load(config['load'])
        model = models.load(model_sv)
    else:
        model = models.make(config['model'], **config['model_args'])

    if eval_fs:
        fs_model = models.make('meta-baseline', encoder=None)
        fs_model.encoder = model.encoder  # use classifier without meta-learning

    if config.get('_parallel'):
        model = nn.DataParallel(model)
        if eval_fs:
            fs_model = nn.DataParallel(fs_model)

    utils.log('num params: {}'.format(utils.compute_n_params(model)))  # model size
    optimizer, lr_scheduler = utils.make_optimizer(model.parameters(), config['optimizer'], **config['optimizer_args'])

    max_epoch = config['max_epoch']
    # save_epoch = config.get('save_epoch')
    max_va = 0.
    timer_used = utils.Timer()
    timer_epoch = utils.Timer()

    # training classifier baseline
    for epoch in range(1, max_epoch + 1 + 1):
        timer_epoch.s()

        aves_keys = ['tl', 'ta', 'vl', 'va']
        if eval_fs:
            for n_shot in n_shots:
                aves_keys += ['fsa-' + str(n_shot)]
        aves = {k: utils.Averager() for k in aves_keys}

        # train classifier baseline
        model.train()
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        for data, label in tqdm(train_loader, desc='train', leave=False):
            data, label = data.cuda(), label.cuda()
            logits = model(data)
            loss = F.cross_entropy(logits, label)
            acc = utils.compute_acc(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            aves['tl'].add(loss.item())  # train loss
            aves['ta'].add(acc)          # train acc

            # logits = None
            # loss = None

        # eval classifier baseline
        if eval_val:
            model.eval()
            for data, label in tqdm(val_loader, desc='val', leave=False):
                data, label = data.cuda(), label.cuda()
                with torch.no_grad():
                    logits = model(data)
                    loss = F.cross_entropy(logits, label)
                    acc = utils.compute_acc(logits, label)
                aves['vl'].add(loss.item())  # val loss
                aves['va'].add(acc)          #  val acc

        # evaluate few-shot learning ability in novel classes 
        if eval_fs and (epoch % ef_epoch == 0 or epoch == max_epoch + 1):
            fs_model.eval()
            for i, n_shot in enumerate(n_shots):
                np.random.seed(0)
                for data, _ in tqdm(fs_loaders[i], desc='fs-' + str(n_shot), leave=False):
                    x_shot, x_query = split_shot_query(data.cuda(), n_way, n_shot, n_query, ep_per_batch=config['ep_per_batch'])
                    label = make_nk_label(n_way, n_query, ep_per_batch=config['ep_per_batch']).cuda()
                    with torch.no_grad():
                        logits = fs_model(x_shot, x_query).view(-1, n_way)
                        acc = utils.compute_acc(logits, label)
                    aves['fsa-' + str(n_shot)].add(acc)

        # post
        if lr_scheduler is not None:
            lr_scheduler.step()

        for k, v in aves.items():
            aves[k] = v.item()

        t_epoch = utils.time_str(timer_epoch.t())
        t_used = utils.time_str(timer_used.t())
        t_estimate = utils.time_str(timer_used.t() / epoch * max_epoch)

        epoch_str = str(epoch)
        log_str = 'epoch {}, train {:.4f}|{:.4f}'.format(epoch_str, aves['tl'], aves['ta'])
        writer.add_scalars('loss', {'train': aves['tl']}, epoch)
        writer.add_scalars('acc', {'train': aves['ta']}, epoch)

        if eval_val:
            log_str += ', val {:.4f}|{:.4f}'.format(aves['vl'], aves['va'])
            writer.add_scalars('loss', {'val': aves['vl']}, epoch)
            writer.add_scalars('acc', {'val': aves['va']}, epoch)

        if eval_fs and (epoch % ef_epoch == 0 or epoch == max_epoch + 1):
            log_str += ', fs'
            for n_shot in n_shots:
                key = 'fsa-' + str(n_shot)
                log_str += ' {}: {:.4f}'.format(n_shot, aves[key])
                writer.add_scalars('acc', {key: aves[key]}, epoch)

        if epoch <= max_epoch:
            log_str += ', {} {}/{}'.format(t_epoch, t_used, t_estimate)
        else:
            log_str += ', {}'.format(t_epoch)
        utils.log(log_str)

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
        
        # if (save_epoch is not None) and epoch % save_epoch == 0:
        #     torch.save(save_obj, os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))

        if aves['va'] > max_va:
            max_va = aves['va']
            torch.save(save_obj, os.path.join(save_path, 'max-va.pth'))
       
        writer.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/train_classifier_mini.yaml')
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

