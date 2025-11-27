import os
import random
import sys

import numpy as np
import torch
import yaml

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

def print_select(rank: int, restriction=True):
    """
    If rank == 0, use default print function to print strings to the console,
    otherwise, do nothing.
    If restriction==False, all ranks can print strings to the console.
    """
    if rank == 0:
        return print
    else:
        if restriction:
            return do_nothing
        else:
            return print

def do_nothing(str, sep=" ", end="\n", file=None, flush=False):
    pass

class Logger(object):
    def __init__(self, log=open("default.log", "a"), stream=sys.stdout):
        self.terminal = stream
        self.log = log
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        pass

class data_prefetcher():
    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream(device)
        self.device = device
        self.preload()
    def preload(self):
        try:
            self.next_batch = next(self.loader)
        except StopIteration:
            self.next_batch = None
            return
        with torch.cuda.stream(self.stream):
            self.next_batch = [t.to(self.device, non_blocking=True) for t in self.next_batch]
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        if batch is not None:
            batch[0].record_stream(torch.cuda.current_stream())
        self.preload()
        return batch

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def load_args(filename, args):
    with open(filename, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    for key, group in data_loaded.items():
        for key, val in group.items():
            setattr(args, key, val)

def compute_params(model, print):
    num_params = 0
    for param in model.parameters():
        if param.requires_grad:
            num_params += param.numel()
    print('Number of model parameters: %.2fM' % (num_params / 1e6))