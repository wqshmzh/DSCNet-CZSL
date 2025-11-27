import os
import random
import sys
import time
from collections import OrderedDict
from multiprocessing import Lock

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.distributed import barrier, init_process_group
from torch.multiprocessing import Process, set_start_method
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import BatchSampler, DistributedSampler
from torch.utils.data.dataloader import DataLoader

import test_utils
from dataset import CompositionDataset, get_dataset_info
from utils import *

torch.backends.cudnn.enabled = False
torch.backends.cuda.matmul.allow_tf32 = False

os.environ["OPENBLAS_NUM_THREADS"] = '4'

from parameters import YML_PATH, parser

config = parser.parse_args()
config.ddp = False
config.device_ids = "0"
config.open_world = False
config.dataset = "ut-zap50k"
config.visualize_features = True
if config.dataset == "ut-zap50k":
    config.model_load = "2024-04-18_18h23m16s_val_best_AUC_46.32.pt"
elif config.dataset == "mit-states":
    config.model_load = "2024-04-20_20h07m49s_val_best_AUC_21.41.pt"
elif config.dataset == "cgqa":
    config.model_load = "2024-03-02_19h06m23s_val_best_AUC_12.15.pt"
elif config.dataset == "clothing16k":
    config.model_load = "2024-05-02_08h17m07s_val_best_AUC_99.09.pt"
elif config.dataset == "ao-clevr":
    config.model_load = "2024-05-02_08h43m59s_val_best_AUC_99.94.pt"
elif config.dataset == "vaw-czsl":
    config.model_load = "2024-05-15_19h55m36s_val_best_AUC_3.11.pt"
os.environ["CUDA_VISIBLE_DEVICES"] = config.device_ids


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
            self.next_batch[0] = self.next_batch[0].to(self.device)
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        if batch is not None:
            batch[0].record_stream(torch.cuda.current_stream())
        self.preload()
        return batch

# t-sne functions
def plot_embedding_new(data, label, title, data2, label2, title2):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    
    _ =  plt.subplot(121)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
#     plt.scatter(data[:, 0], data[:, 1], c=label, cmap=plt.cm.Spectral)

    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    
    x_min2, x_max2 = np.min(data2, 0), np.max(data2, 0)
    data2 = (data2 - x_min2) / (x_max2 - x_min2)
        
#     fig = plt.figure()
    _ = plt.subplot(122)
    for i in range(data2.shape[0]):
        plt.text(data2[i, 0], data2[i, 1], str(label2[i]),
                 color=plt.cm.Set1(label2[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([])
    plt.yticks([])
    plt.title(title2)

    return fig

# obtain_data
def obtain_data(features_primitive,primitives,topk,num_per_class,primitive2idx,pri_name):
    
    if len(primitives) > topk:
        primitives = primitives[:topk]
    
    primitive_data = np.zeros((len(primitives)*num_per_class,768))
#     primitives = list(primitive2idx.keys())[:topk]
    primitive_label = []
    count = 0 
    for i,primitive in enumerate(primitives):
        assert len(primitive2idx[primitive]) > num_per_class
        selected_indices = random.sample(primitive2idx[primitive],num_per_class)
        for idx in selected_indices:
            primitive_data[count,:] = features_primitive[idx]
            if pri_name == 'att':
#                 primitive_label.append(0 + i)
                primitive_label.append(0 + i)
            else:
                primitive_label.append(topk + i )
            count += 1
            
    return primitive_data, primitive_label

def evaluate(config, model, loader, attrs, objs, print):
    dataset = loader.dataset
    if dataset.phase == "val":
        print("--Evaluating on validation set-- ")
    elif dataset.phase == "test":
        print("--Evaluating on testing set-- ")
    model.eval()
    
    attr2idx = dataset.attr2idx
    obj2idx = dataset.obj2idx
    pairs_dataset = dataset.pairs
    pairs = torch.tensor([(attr2idx[attr], obj2idx[obj])
                                for attr, obj in pairs_dataset]).cuda()
    prefetcher = data_prefetcher(loader, config.device)
    progress_bar = range(len(loader))
    
    try:
        attr_predicts = torch.load(os.path.join(config.save_path, "attr_predicts.pt"))
        obj_predicts = torch.load(os.path.join(config.save_path, "obj_predicts.pt"))
    except:
        raise FileNotFoundError("Predicted attributes or objects not found locally. Please run evaluation.py first.")
    
    all_attr_vis_embs, all_attr_text_embs, all_obj_vis_embs, all_obj_text_embs = {}, {}, {}, {}
    all_attr_con_vis_embs, all_attr_con_text_embs, all_obj_con_vis_embs, all_obj_con_text_embs = {}, {}, {}, {}
    
    for i in range(len(attrs)):
        all_attr_vis_embs[attrs[i]] = []
        all_obj_vis_embs[objs[i]] = []
        all_attr_text_embs[attrs[i]] = []
        all_obj_text_embs[objs[i]] = []
        all_attr_con_vis_embs[attrs[i]] = []
        all_obj_con_vis_embs[objs[i]] = []
        all_attr_con_text_embs[attrs[i]] = []
        all_obj_con_text_embs[objs[i]] = []
    
    with torch.no_grad():
        for idx in progress_bar:
            batch = prefetcher.next()
            img_proj_attr, img_proj_obj, attr_prompt_embs, obj_prompt_embs, \
            conditional_attr_img_embs, conditional_obj_img_embs, conditional_attr_embs, conditional_obj_embs = model(batch, pairs)
            img_id_attr_i = []
            img_id_obj_i = []
            for i in range(len(attrs)):
                for j in range(len(batch[0])):
                    if attr_predicts[j] == dataset.attr2idx[attrs[i]]:
                        img_id_attr_i.append(j)
                    if obj_predicts[j] == dataset.obj2idx[objs[i]]:
                        img_id_obj_i.append(j)
                img_id_attr_i = torch.tensor(img_id_attr_i).cuda()
                img_id_obj_i = torch.tensor(img_id_obj_i).cuda()
                all_attr_vis_embs[attrs[i]].append(img_proj_attr[img_id_attr_i].cpu())
                all_obj_vis_embs[objs[i]].append(img_proj_obj[img_id_obj_i].cpu())
                all_attr_text_embs[attrs[i]].append(attr_prompt_embs[img_id_attr_i].cpu())
                all_obj_text_embs[objs[i]].append(obj_prompt_embs[img_id_obj_i].cpu())
                all_attr_con_vis_embs[attrs[i]].append(conditional_attr_img_embs[img_id_attr_i].cpu())
                all_obj_con_vis_embs[objs[i]].append(conditional_obj_img_embs[img_id_obj_i].cpu())
                all_attr_con_text_embs[attrs[i]].append(conditional_attr_embs[img_id_attr_i].cpu())
                all_obj_con_text_embs[objs[i]].append(conditional_obj_embs[img_id_obj_i].cpu())
    
    for i in range(len(attrs)):
        all_attr_vis_embs[attrs[i]] = torch.cat(all_attr_vis_embs[attrs[i]]).numpy()
        all_obj_vis_embs[objs[i]] = torch.cat(all_obj_vis_embs[objs[i]]).numpy()
        all_attr_text_embs[attrs[i]] = torch.cat(all_attr_text_embs[attrs[i]]).numpy()
        all_obj_text_embs[objs[i]] = torch.cat(all_obj_text_embs[objs[i]]).numpy()
        all_attr_con_vis_embs[attrs[i]] = torch.cat(all_attr_con_vis_embs[attrs[i]]).numpy()
        all_obj_con_vis_embs[objs[i]] = torch.cat(all_obj_con_vis_embs[objs[i]]).numpy()
        all_attr_con_text_embs[attrs[i]] = torch.cat(all_attr_con_text_embs[attrs[i]]).numpy()
        all_obj_con_text_embs[objs[i]] = torch.cat(all_obj_con_text_embs[objs[i]]).numpy()
                        

if __name__ == "__main__":
    rank = 0
    torch.cuda.set_device(rank)
    print = print_select(rank) # Only rank0 gets real print function, other ranks get "do_nothing" function
    config.rank = rank
    config.device = f"cuda:{config.rank}"
    torch.cuda.set_device(config.device)
    config.clip_path = "/root/pretrained_models/CLIP"
    config.dataset_root = "/root/datasets/"
    config.model = "full"
    config.seed = 1
    if not config.port:
        config.port = 2346
    load_args(YML_PATH[config.dataset], config)
    config.save_path = os.path.join(config.main_root, "data", f"{config.model}", config.dataset)

    if config.ddp: init_process_group(backend="nccl", init_method=f"tcp://localhost:{config.port}", world_size=config.world_size, rank=config.rank)

    device_info = torch.cuda.get_device_properties(config.rank)
    print(f"Choose device: {config.device} ({device_info.name}, {round(device_info.total_memory/1024**3, ndigits=2)}GB)")

    print(config)

    # set the seed value
    set_seed(config.seed)
    
    # load datasets
    dataset_path = config.dataset_root + config.dataset
    train_data, val_data, test_data, all_attrs, all_objs, \
        all_pairs, tr_pairs, vl_pairs, ts_pairs = get_dataset_info(root=dataset_path, 
                                                                   split="compositional-split-natural",
                                                                   print=print)
    # val_dataset = CompositionDataset(dataset_path,
    #                                  config=config,
    #                                  phase="val",
    #                                  split="compositional-split-natural",
    #                                  data=val_data,
    #                                  all_attrs=all_attrs,
    #                                  all_objs=all_objs,
    #                                  all_pairs=all_pairs,
    #                                  train_pairs=tr_pairs,
    #                                  split_pairs=vl_pairs,
    #                                  open_world=config.open_world,
    #                                  print=print)
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
    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=config.eval_batch_size,
    #     shuffle=False,
    #     pin_memory=True,
    #     num_workers=config.num_workers,
    #     persistent_workers=True,
    #     prefetch_factor=10
    # )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=config.num_workers,
        persistent_workers=True,
        prefetch_factor=10,
    )

    objects = [cla.replace(".", " ").lower() for cla in all_objs]
    attributes = [attr.replace(".", " ").lower() for attr in all_attrs]
    offset = len(attributes)

    # load model
    from model.full_model import CSNet
    model = CSNet(config, attributes=attributes, objects=objects, offset=offset).cuda()

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
    
    # init ddp model
    if config.ddp:
        model = DistributedDataParallel(module=model, device_ids=[config.rank], output_device=config.rank, find_unused_parameters=False)
    
    # print hyperparameters
    print("Hyper parameters:")
    print(f"  total evaluation batch size = {config.eval_batch_size}")
    
    attrs = ["Canvas", "Satin", "Synthetic"]
    objs = ["Shoes.Flats", "Shoes.Clogs.and.Mules", 'Boots.Mid.Calf']
    torch.cuda.empty_cache()
    original_features, conditional_features = evaluate(config, model, test_loader, attrs, objs, print)
    print("------------evaluation end------------")
    
    ''' 画图 ''' 

    topk = 3
    num_per_class = 20

    legend_attrs = attrs + objs

    data = np.concatenate((att_data,obj_data), axis=0)
    label = np.concatenate((att_label,obj_label), axis=0)
    # label = (label>=topk).astype(int)

    # data, label, n_samples, n_features = get_data()
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    result = tsne.fit_transform(data)
    # fig = plot_embedding(result, label, dataset)
    # fig = plot_embedding_new(result, label, dataset)

    # plt.subplot(1, 2, 1)
    # plt.show(fig)

    data2 = np.concatenate((att_data2,obj_data2), axis=0)
    label2 = np.concatenate((att_label2,obj_label2), axis=0)
    # label2 = (label2>=topk).astype(int)

    tsne = TSNE(n_components=2, init='pca', random_state=0)
    result2 = tsne.fit_transform(data2)
    # fig2 = plot_embedding(result2, label2, dataset)

    fig = plot_embedding_new(result, label, 'Ours', result2, label2, 'Ours wo RM')

    plt.legend(handles=[l1,l2],labels=legend,loc='best')


    plt.show(fig)

    # # plt.show(fig)
    # _ = plt.subplot(1, 2, 2)
    # _ = plt.show(fig2)

    plt.legend()
