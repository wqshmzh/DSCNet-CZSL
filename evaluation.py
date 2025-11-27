from parameters import YML_PATH, parser

config = parser.parse_args()
config.device_ids = "0"

import os

os.environ["CUDA_VISIBLE_DEVICES"] = config.device_ids
import sys
import time
from multiprocessing import Lock

import torch
from torch.distributed import barrier, init_process_group
from torch.multiprocessing import Process, set_start_method
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import BatchSampler, DistributedSampler
from torch.utils.data.dataloader import DataLoader

import test_utils
from dataset import CompositionDataset, get_dataset_info
from model.DSCNet import DSCNet
from utils import *

torch.backends.cudnn.enabled = False
torch.backends.cuda.matmul.allow_tf32 = False

os.environ["OPENBLAS_NUM_THREADS"] = '4'

def main(rank, config, **kwargs):
    torch.cuda.set_device(rank)
    print = print_select(rank) # Only rank0 gets real print function, other ranks get "do_nothing" function
    config.open_world = False
    config.dataset = "ut-zap50k"
    config.model_load = config.dataset + ".pt"
    config.rank = rank
    config.device = f"cuda:{config.rank}"
    torch.cuda.set_device(config.device)
    config.seed = 1
    load_args(YML_PATH["general"], config)
    load_args(YML_PATH[config.dataset], config)
    config.eval_batch_size = 1024
    
    time_str = time.strftime("%Y-%m-%d_%Hh%Mm%Ss")
    log_file_name = f"log_{time_str}.txt"
    config.save_path = os.path.join(config.main_root, "saves", config.dataset)
    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path, exist_ok=True)

    device_info = torch.cuda.get_device_properties(config.rank)
    print(f"Choose device: {config.device} ({device_info.name}, {round(device_info.total_memory/1024**3, ndigits=2)}GB)")

    print(config)

    # set the seed value
    set_seed(config.seed)
    
    # load datasets
    dataset_path = config.dataset_root + config.dataset
    _, _, test_data, all_attrs, all_objs, \
        all_pairs, tr_pairs, _, ts_pairs = get_dataset_info(root=dataset_path, 
                                                                   split="compositional-split-natural",
                                                                   print=print)
    test_dataset = CompositionDataset(dataset_path,
                                      config=config,
                                      phase="test",
                                      split="compositional-split-natural",
                                      data=test_data,
                                      all_attrs=all_attrs,
                                      all_objs=all_objs,
                                      all_pairs=all_pairs,
                                      train_pairs=tr_pairs,
                                      split_pairs=ts_pairs,
                                      open_world=config.open_world,
                                      print=print)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        pin_memory=config.pin_memory,
        num_workers=config.num_workers,
        persistent_workers=True,
        prefetch_factor=10,
    )

    objects = [cla.replace(".", " ").lower() for cla in all_objs]
    attributes = [attr.replace(".", " ").lower() for attr in all_attrs]
    offset = len(attributes)

    # load model
    model = DSCNet(config, attributes=attributes, objects=objects, offset=offset).cuda()

    # Load trained model
    try:
        print("Load checkpoint")
        checkpoint = torch.load(os.path.join(config.save_path, config.model_load), map_location=config.device)
        for param_name, param in checkpoint.items():
            if "module." in param_name:
                param_name = param_name.replace("module.", "")
                checkpoint[param_name] = param
            else:
                break
        model.load_state_dict(checkpoint, strict=False)
    except:
        print("No saved model found in local disk. \n[NOT RECOMMENDED] The model will be evaluated with randomly initialized parameters.")
    
    # print hyperparameters
    print("Hyper parameters:")
    print(f"  total evaluation batch size = {config.eval_batch_size * config.world_size}")
    
    torch.cuda.empty_cache()
    test(config, model, test_loader, None, None, None, print, time_str)
    print("------------evaluation end------------")

def timer(elapsed_time):
    days = int(elapsed_time / (3600 * 24))
    hours = int((elapsed_time - 3600 * 24 * days) / 3600)
    minutes = int((elapsed_time - 3600 * 24 * days - 3600 * hours) / 60)
    seconds = int(elapsed_time - 3600 * 24 * days - 3600 * hours - 60 * minutes)
    miliseconds = int(1000 * (elapsed_time - 3600 * 24 * days - 3600 * hours - 60 * minutes - seconds))
    return days, hours, minutes, seconds, miliseconds    

def test(config, model, val_loader, best_metric, metrics, epoch, print, time_str):
    print("--Evaluating on testing set-- ")
    _ = evaluate(config, model, val_loader, print)

def evaluate(config, model, loader, print):
    start_time = time.time()
    dataset = loader.dataset
    model.eval()
    
    evaluator = test_utils.Evaluator(dataset)
    all_logits, all_attr_gt, all_obj_gt, all_pair_gt = test_utils.predict_logits(
            model, loader, config)
    
    if config.open_world:
        all_logits = test_utils.filter_logits_with_feasibility(
            logits=all_logits,
            seen_or_feasible_mask=dataset.seen_or_feasible_mask
            )
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
    print("Attr Acc: {:.2f}% | Obj Acc: {:.2f}% | Seen Acc: {:.2f}% | Unseen Acc: {:.2f}% | HM: {:.2f}% | AUC: {:.2f} | {}hours: {}min: {}sec: {}ms".\
        format(test_stats["attr_acc"]*100, test_stats["obj_acc"]*100, test_stats["best_seen"]*100, test_stats["best_unseen"]*100, test_stats["best_hm"]*100, test_stats["AUC"]*100, hours, minutes, seconds, miliseconds))
    
    return test_stats

if __name__ == "__main__":
    if not config.ddp:
        config.world_size = 1
        main(0, config)
    else:
        set_start_method("spawn")
        lock = Lock()
        processes = []
        config.world_size = len(config.device_ids.split(","))
        for rank in range(config.world_size):
            p = Process(target=main, 
                        args=(rank, config), 
                        kwargs={"lock": lock})
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    print("Program terminated")
