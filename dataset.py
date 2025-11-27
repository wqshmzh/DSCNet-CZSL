import re
from itertools import product

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import (CenterCrop, ColorJitter, Compose,
                                    InterpolationMode, Normalize,
                                    RandomHorizontalFlip, RandomPerspective,
                                    RandomRotation, Resize, ToTensor)
from torchvision.transforms.transforms import RandomResizedCrop

BICUBIC = InterpolationMode.BICUBIC
n_px = 224


def get_norm_values(norm_family):
    '''
        Inputs
            norm_family: String of norm_family
        Returns
            mean, std : tuple of 3 channel values
    '''
    if 'resnet' in norm_family:
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    elif norm_family == 'clip':
        mean, std = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
    else:
        raise ValueError('Incorrect normalization family')
    return mean, std

def transform_image(split="train", norm_family='clip'):
    mean, std = get_norm_values(norm_family=norm_family)
    if split == "test" or split == "val":
        transform = Compose(
            [
                Resize((n_px, n_px), interpolation=BICUBIC),
                CenterCrop((n_px, n_px)),
                ToTensor(),
                Normalize(mean, std),
            ]
        )
    else:
        transform = Compose(
            [
                Resize((n_px, n_px), interpolation=BICUBIC),
                CenterCrop((n_px, n_px)),
                RandomHorizontalFlip(),
                RandomRotation(degrees=5, interpolation=BICUBIC),
                ToTensor(),
                Normalize(mean, std),
            ]
        )

    return transform

class ImageLoader:
    def __init__(self, root):
        self.img_dir = root

    def __call__(self, img):
        if 'mit-states' in self.img_dir:
            pair, img_name = img.split('/')
            img = pair.replace('_', ' ') + '/' + img_name
        file = '%s/%s' % (self.img_dir, img)
        img = Image.open(file).convert('RGB')
        return img


def get_dataset_info(root, split, print=print):
    def parse_pairs(pair_list):
        with open(pair_list, 'r') as f:
            pairs = f.read().strip().split('\n')
            pairs = [re.split('[ |+]', t) for t in pairs]
            pairs = list(map(tuple, pairs))
        attrs, objs = zip(*pairs)
        return attrs, objs, pairs

    tr_attrs, tr_objs, tr_pairs = parse_pairs(
        '%s/%s/train_pairs.txt' % (root, split))
    vl_attrs, vl_objs, vl_pairs = parse_pairs(
        '%s/%s/val_pairs.txt' % (root, split))
    ts_attrs, ts_objs, ts_pairs = parse_pairs(
        '%s/%s/test_pairs.txt' % (root, split))

    all_attrs, all_objs = sorted(
        list(set(tr_attrs + vl_attrs + ts_attrs))), sorted(
            list(set(tr_objs + vl_objs + ts_objs)))
    all_pairs = sorted(list(set(tr_pairs + vl_pairs + ts_pairs)))
    
    data = torch.load(root + '/metadata_{}.t7'.format(split))
    train_data, val_data, test_data = [], [], []
    for instance in data:
        image, attr, obj, settype = instance['image'], \
            instance['attr'], instance['obj'], instance['set']

        if attr == 'NA' or (attr, obj) not in all_pairs or settype == 'NA':
            # ignore instances with unlabeled attributes
            # ignore instances that are not in current split
            continue

        if settype == 'train':
            train_data.append([image, attr, obj])
        elif settype == 'val':
            val_data.append([image, attr, obj])
        elif settype == 'test':
            test_data.append([image, attr, obj])
    
    print('# All        - images: %d | attributes: %d | objects: %d | pairs: %d | open world pairs: %d' % 
                  (len(train_data+val_data+test_data), len(all_attrs), len(all_objs), len(all_pairs), len(all_attrs)*len(all_objs)))

    return train_data, val_data, test_data, all_attrs, all_objs, all_pairs, tr_pairs, vl_pairs, ts_pairs


class CompositionDataset(Dataset):
    def __init__(
            self,
            root,
            config,
            phase,
            data,
            all_attrs,
            all_objs,
            all_pairs,
            train_pairs,
            split_pairs,
            split='compositional-split-natural',
            open_world=False,
            print=print
    ):
        self.root = root
        self.phase = phase
        self.data = data
        self.attrs = all_attrs
        self.objs = all_objs
        self.pairs = all_pairs
        self.train_pairs = train_pairs
        self.split_pairs = split_pairs
        self.split = split
        self.open_world = open_world
        self.feat_dim = None
        self.transform = transform_image(phase, config.norm_family)
        self.loader = ImageLoader(self.root + '/images')

        self.obj2idx = {obj: idx for idx, obj in enumerate(all_objs)}
        self.attr2idx = {attr: idx for idx, attr in enumerate(all_attrs)}
        
        if self.phase == 'train':
            self.data = data
            print('# Training   - images: %d | pairs: %d' % (len(data), len(train_pairs)))
            self.train_pair_to_idx = dict(
                [(pair, idx) for idx, pair in enumerate(train_pairs)]
                )
        else:
            self.data = data
            image_mask = torch.BoolTensor([1 if (data[1], data[2]) in train_pairs else 0 for data in data])
            pair_mask = torch.BoolTensor([1 if pair in train_pairs else 0 for pair in split_pairs]) # 1 for seen pair, 0 for unseen pair
            num_seen_images = int(image_mask.sum())
            num_seen_pairs = int(pair_mask.sum())
            if phase == 'val':
                print('# Validation - images: %d | pairs: %d | seen images: %d | seen pairs: %d | unseen images: %d | unseen pairs: %d' % 
                      (len(data), len(split_pairs), num_seen_images, num_seen_pairs, len(data)-num_seen_images, len(split_pairs)-num_seen_pairs))
            elif phase == 'test':
                print('# Testing    - images: %d | pairs: %d | seen images: %d | seen pairs: %d | unseen images: %d | unseen pairs: %d' % 
                      (len(data), len(split_pairs), num_seen_images, num_seen_pairs, len(data)-num_seen_images, len(split_pairs)-num_seen_pairs))
            else:
                NotImplementedError("Unrecognized phase. Please choose a phase from train, val, and test")

        if self.open_world:
            import os
            from test_utils import filter_pairs_with_feasibility
            from model.feasibility import compute_feasibility
            
            self.obj_by_attrs_train = {k: [] for k in self.attrs}
            for (a, o) in self.train_pairs:
                self.obj_by_attrs_train[a].append(o)

            # Intantiate attribut-object relations, needed just to evaluate mined pairs
            self.attrs_by_obj_train = {k: [] for k in self.objs}
            for (a, o) in self.train_pairs:
                self.attrs_by_obj_train[o].append(a)
            
            self.all_pairs = list(product(all_attrs, all_objs))
            all_pair2idx = {pair: idx for idx, pair in enumerate(self.all_pairs)}
            mask = [1 if pair in set(self.train_pairs) else 0 for pair in self.all_pairs]
            self.seen_mask = torch.BoolTensor(mask) * 1.
            mask = [1 if pair in set(self.train_pairs + self.split_pairs) else 0 for pair in self.all_pairs]
            self.closed_world_mask = torch.BoolTensor(mask) * 1.
            
            feasibility_file = os.path.join(config.main_root, "saves", "feasibility_"+config.dataset+".pt")
            if not os.path.exists(feasibility_file):
                feasibility = compute_feasibility(all_objs, all_attrs, self.seen_mask, train_pairs, all_pair2idx, self.attrs_by_obj_train, self.obj_by_attrs_train, config.glove_root)
                torch.save({'feasibility': feasibility,}, feasibility_file)
            self.feasibility = torch.load(f=feasibility_file)["feasibility"]
            self.pairs, self.feasibility_mask = filter_pairs_with_feasibility(self.all_pairs, self.closed_world_mask, config.threshold, self.feasibility)
            
            feasible_pairs = []
            for feasibility_id in range(len(self.feasibility_mask)):
                if self.feasibility_mask[feasibility_id] > 0:
                    feasible_pairs.append(self.all_pairs[feasibility_id])
            seen_or_feasible_mask = [1 if pair in set(self.train_pairs + feasible_pairs) else 0 for pair in self.pairs]
            self.seen_or_feasible_mask = torch.BoolTensor(seen_or_feasible_mask) * 1.

        self.pair2idx = {pair: idx for idx, pair in enumerate(self.pairs)}

    def __getitem__(self, index):
        image, attr, obj = self.data[index]
        img = self.loader(image)
        img = self.transform(img)

        if self.phase == 'train':
            data = [
                img, self.attr2idx[attr], self.obj2idx[obj], self.train_pair_to_idx[(attr, obj)]
            ]
        else:
            data = [
                img, self.attr2idx[attr], self.obj2idx[obj], self.pair2idx[(attr, obj)]
            ]

        return data

    def __len__(self):
        return len(self.data)
