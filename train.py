import os

from parameters import YML_PATH, parser

config = parser.parse_args()
config.device_ids = '0'
os.environ['OPENBLAS_NUM_THREADS'] = "8"
try:
    os.environ['AMDGPU_TARGETS']
    os.environ['HIP_VISIBLE_DEVICES'] = config.device_ids
except:
    os.environ['CUDA_VISIBLE_DEVICES'] = config.device_ids

import collections
import sys
import time
from os.path import join as ospj

import numpy as np
import torch
from torch.amp import GradScaler
from torch.amp import autocast as autocast
from torch.multiprocessing import set_start_method
from torch.utils.data.dataloader import DataLoader

import test_utils as test_utils
from dataset import CompositionDataset, get_dataset_info
from model.DSCNet import DSCNet
from utils import *


def main(rank, config, **kwargs):
    torch.cuda.set_device(rank)
    print = print_select(rank) # Only rank0 gets real print function, other ranks get "do_nothing" function
    config.persistent_workers = True
    config.pin_memory = True
    config.rank = rank
    config.device = f'cuda:{config.rank}'
    config.open_world = False
    config.seed = 1
    config.dataset = 'ut-zap50k'
    config.model_load = 'checkpoint_latest.pt'
    
    load_args(YML_PATH[config.dataset], config)
    load_args(YML_PATH["general"], config)

    config.train_batch_size = int(config.train_batch_size // config.gradient_accumulation_steps)
    
    print('max epochs: ', config.epochs)

    if config.save_model:
        time_str = time.strftime('%Y-%m-%d_%Hh%Mm%Ss')
        log_file_name = f'log_{time_str}.txt'
        config.save_path = os.path.join(os.path.dirname(__file__), 'data', config.dataset)
        if not os.path.exists(config.save_path):
            os.makedirs(config.save_path, exist_ok=True)
        log_file = open(os.path.join(config.save_path, log_file_name), 'a')
        text_logger = Logger(log_file)
        sys.stdout = text_logger
        sys.stderr = text_logger

    device_info = torch.cuda.get_device_properties(config.rank)
    print(f'Choose device: {config.device} ({device_info.name}, {round(device_info.total_memory/1024**3, ndigits=2)}GB)')

    print(config)

    # set the seed value
    set_seed(config.seed)
    
    # load datasets
    dataset_path = config.dataset_root + config.dataset
    train_data, val_data, test_data, all_attrs, all_objs, \
        all_pairs, tr_pairs, vl_pairs, ts_pairs = get_dataset_info(root=dataset_path, 
                                                                   split='compositional-split-natural',
                                                                   print=print)
    train_dataset = CompositionDataset(dataset_path,
                                       config=config,
                                       phase='train',
                                       split='compositional-split-natural',
                                       data=train_data,
                                       all_attrs=all_attrs,
                                       all_objs=all_objs,
                                       all_pairs=all_pairs,
                                       train_pairs=tr_pairs,
                                       split_pairs=tr_pairs,
                                       print=print)
    val_dataset = CompositionDataset(dataset_path,
                                     config=config,
                                     phase='val',
                                     split='compositional-split-natural',
                                     data=val_data,
                                     all_attrs=all_attrs,
                                     all_objs=all_objs,
                                     all_pairs=all_pairs,
                                     train_pairs=tr_pairs,
                                     split_pairs=vl_pairs,
                                     print=print)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        pin_memory=config.pin_memory,
        num_workers=config.num_workers,
        persistent_workers=config.persistent_workers,
        prefetch_factor=10
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        pin_memory=config.pin_memory,
        num_workers=config.num_workers,
        persistent_workers=config.persistent_workers,
        prefetch_factor=10
    )

    objects = [obj.replace(".", " ").lower() for obj in all_objs]
    attributes = [attr.replace(".", " ").lower() for attr in all_attrs]
    offset = len(attributes)
    
    model = DSCNet(config, attributes=attributes, objects=objects, offset=offset).cuda()
    compute_params(model, print)

    # init optimizer
    optim_func = torch.optim.Adam
    optimizer = optim_func(
        params=[{'name': name, 'params': param} for name, param in dict(model.named_parameters()).items() if param.requires_grad],
        lr=config.lr, weight_decay=config.weight_decay
        )
    optimizer.zero_grad()
    metrics = {
        'best_attr': 0.0,
        'best_obj': 0.0,
        'best_seen': 0.0,
        'best_unseen': 0.0,
        'best_hm': 0.0,
        'best_auc': 0.0,
        'best_epoch': 0.0
    }

    # Load trained model
    try:
        checkpoint = torch.load(os.path.join(config.save_path, config.model_load), map_location=config.device)
        print(config.model_load)
        print('Load checkpoint')
        checkpoint = torch.load(ospj(config.save_path, config.model_load), map_location=config.device)
        model.load_state_dict(checkpoint['net'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        config.epoch_start = checkpoint['epoch'] + 1
        ckpt_metrics = checkpoint['metrics']
        if config.rank == 0:
            metrics['best_attr'] = ckpt_metrics['best_attr']
            metrics['best_obj'] = ckpt_metrics['best_obj']
            metrics['best_seen'] = ckpt_metrics['best_seen']
            metrics['best_unseen'] = ckpt_metrics['best_unseen']
            metrics['best_auc'] = ckpt_metrics['best_AUC']
            metrics['best_hm'] = ckpt_metrics['best_HM']
            metrics['best_epoch'] = ckpt_metrics['best_epoch']
    except:
        print('No saved model found in local disk. The model will be trained from scratch')
        config.epoch_start = 0
    
    # init lr scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5, last_epoch=config.epoch_start-1)
    
    # print hyperparameters
    print('Hyper parameters:')
    print(f'  total training batch size = {config.train_batch_size * config.gradient_accumulation_steps}')
    print(f'  total evaluation batch size = {config.eval_batch_size}')

    torch.cuda.empty_cache()
    
    best_metric = 0
    train_losses = []
    if config.use_amp:
        scaler = GradScaler()
    else:
        scaler = None
    for epoch in range(config.epoch_start, config.epochs):
        print('Epoch {} | Best Attr: {:.2f}% | Best Obj: {:.2f}% | Best Seen: {:.2f}% | Best Unseen: {:.2f}% | Best HM: {:.2f}% | Best AUC: {:.2f} | Best Epoch: {:.0f}'.\
            format(epoch+1, metrics['best_attr']*100, metrics['best_obj']*100, metrics['best_seen']*100, metrics['best_unseen']*100, metrics['best_hm']*100, 
                   metrics['best_auc']*100, metrics['best_epoch']))
        train_model(epoch, model, optimizer, scheduler, config, train_loader, train_losses, metrics, print, scaler=scaler)
        torch.cuda.empty_cache()
        best_metric = validate(config, model, val_loader, best_metric, metrics, epoch, print, time_str)
        print('------------epoch end------------')
        torch.cuda.empty_cache()

def timer(elapsed_time):
    days = int(elapsed_time / (3600 * 24))
    hours = int((elapsed_time - 3600 * 24 * days) / 3600)
    minutes = int((elapsed_time - 3600 * 24 * days - 3600 * hours) / 60)
    seconds = int(elapsed_time - 3600 * 24 * days - 3600 * hours - 60 * minutes)
    miliseconds = int(1000 * (elapsed_time - 3600 * 24 * days - 3600 * hours - 60 * minutes - seconds))
    return days, hours, minutes, seconds, miliseconds    

def train_model(epoch, model, optimizer, scheduler, config, train_loader, train_losses, metrics, print, **kwargs):
    start_time = time.time()
    model.train()
    model.clip.eval()

    attr2idx = train_loader.dataset.attr2idx
    obj2idx = train_loader.dataset.obj2idx

    train_pairs = torch.tensor([(attr2idx[attr], obj2idx[obj])
                                for attr, obj in train_loader.dataset.train_pairs]).cuda()
                                
    train_prefetcher = data_prefetcher(train_loader, config.device)

    progress_bar = range(len(train_loader))
    epoch_train_losses = []
    for bid in progress_bar:
        batch = train_prefetcher.next()
        with autocast(device_type="cuda", enabled=config.use_amp):
            loss = model(batch, train_pairs) / config.gradient_accumulation_steps

        if ((bid + 1) % config.gradient_accumulation_steps == 0) or (bid + 1 == len(train_loader)):
            is_update = True
        else:
            is_update = False
        
        current_lr = scheduler.get_last_lr()[0]
        # weights update
        if is_update:
            if config.use_amp:
                kwargs['scaler'].scale(loss).backward()
                kwargs['scaler'].step(optimizer)
                kwargs['scaler'].update()
            else:
                loss.backward()
                optimizer.step()
            optimizer.zero_grad()
        else:
            if config.use_amp:
                kwargs['scaler'].scale(loss).backward()
            else:
                loss.backward()
        epoch_train_losses.append(loss.item())
    
    scheduler.step()
    epoch_loss = np.mean(epoch_train_losses)
    days, hours, minutes, seconds, miliseconds = timer(time.time() - start_time)
    print('----Train Lr: {:.4e} | Loss: {:.4e} | {}days: {}hours: {}min: {}sec: {}ms'.format(current_lr, epoch_loss, days, hours, minutes, seconds, miliseconds))

    if config.save_model and config.rank == 0 and (epoch+1) % config.save_every_n == 0:
        print('--Saving training checkpoint--')
        params = model.module.state_dict() if config.ddp else model.state_dict()
        torch.save({'net_param': collections.OrderedDict(
            {param[0]: param[1] for param in params.items() if 'clip' not in param[0] and 'resnet' not in param[0]}
            ),
                    'optimizer': optimizer.state_dict(),
                    'metrics': metrics,
                    'epoch': epoch},
                    os.path.join(config.save_path, f"checkpoint_latest.pt"))

def validate(config, model, val_loader, best_metric, metrics, epoch, print, time_str):
    print("--Evaluating on validation set-- ", end='')
    val_result = evaluate(config, model, val_loader, print)
    if val_result[config.best_model_metric] > best_metric:
        metrics['best_attr'] = val_result['attr_acc']
        metrics['best_obj'] = val_result['obj_acc']
        metrics['best_seen'] = val_result['best_seen']
        metrics['best_unseen'] = val_result['best_unseen']
        metrics['best_hm'] = val_result['best_hm']
        metrics['best_auc'] = val_result['AUC']
        metrics['best_epoch'] = epoch+1
        best_metric = val_result[config.best_model_metric]
        if config.save_model:
            best_metric_save = round(best_metric*100, 2)
            print('--Got higher result. Saving model--')
            params = model.module.state_dict() if config.ddp else model.state_dict()
            # delete duplicate parameter file and only save the latest parameter file with the highest AUC
            for saved_file_name in os.listdir(config.save_path):
                if f'{time_str}_val_best_{config.best_model_metric}' in saved_file_name:
                    os.remove(os.path.join(config.save_path, saved_file_name))
            torch.save(collections.OrderedDict(
                    {param[0]: param[1] for param in params.items() if 'clip' not in param[0] and 'renset' not in param[0]}
                    ), os.path.join(config.save_path, f'{time_str}_val_best_{config.best_model_metric}_{best_metric_save}.pt'))
    return best_metric

def evaluate(config, model, loader, print):
    start_time = time.time()
    dataset = loader.dataset
    model.eval()
    evaluator = test_utils.Evaluator(dataset)
    all_logits, all_attr_gt, all_obj_gt, all_pair_gt = test_utils.predict_logits(
            model, loader, config)
    test_stats = test_utils.test(
            dataset,
            evaluator,
            all_logits,
            all_attr_gt,
            all_obj_gt,
            all_pair_gt,
            config
        )
    days, hours, minutes, seconds, miliseconds = timer(time.time() - start_time)
    print('Attr Acc: {:.2f}% | Obj Acc: {:.2f}% | Seen Acc: {:.2f}% | Unseen Acc: {:.2f}% | HM: {:.2f}% | AUC: {:.2f} | {}hours: {}min: {}sec: {}ms'.\
        format(test_stats['attr_acc']*100, test_stats['obj_acc']*100, test_stats['best_seen']*100, test_stats['best_unseen']*100, test_stats['best_hm']*100, test_stats['AUC']*100, hours, minutes, seconds, miliseconds))
    return test_stats

if __name__ == "__main__":
    config.ddp = False
    set_start_method('spawn')
    main(0, config)
    print('Program terminated')