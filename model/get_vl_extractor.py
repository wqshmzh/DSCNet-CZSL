import os
import torch
from word_embeddings import load_word_embeddings

def get_vl_extractor(config, **kwargs):
    if config.vl_extractor == "clip":
        from model.clip_modules.model_loader import load
        return load(config.clip_model, context_length=config.context_length, download_root=config.clip_path)[0]
    elif "resnet18" in config.vl_extractor:
        from model.resnet18 import ResNet18
        ve = ResNet18()
        attr_word_emb_file = '{}_{}_attr.save'.format(config.dataset, config.emb_type)
        attr_word_emb_file = os.path.join(config.main_root, 'word embedding', attr_word_emb_file)
        obj_word_emb_file = '{}_{}_obj.save'.format(config.dataset, config.emb_type)
        obj_word_emb_file = os.path.join(config.main_root, 'word embedding', obj_word_emb_file)
        print('Load attribute word embeddings--')
        if os.path.exists(attr_word_emb_file):
            pretrained_weight_attr = torch.load(attr_word_emb_file, map_location=config.device)
        else:
            pretrained_weight_attr = load_word_embeddings(kwargs['attrs'], config, print)
            print('Save attr word embeddings using {}'.format(config.emb_type))
            torch.save(pretrained_weight_attr, attr_word_emb_file)        
        print('Load object word embeddings--')
        if os.path.exists(obj_word_emb_file):
            pretrained_weight_obj = torch.load(obj_word_emb_file, map_location=config.device)
        else:
            pretrained_weight_obj = load_word_embeddings(kwargs['objs'], config, print)
            print('Save obj word embeddings using {}'.format(config.emb_type))
            torch.save(pretrained_weight_obj, obj_word_emb_file)
        return ve, pretrained_weight_attr, pretrained_weight_obj
    else:
        NotImplementedError("Illegal choice of extractor")