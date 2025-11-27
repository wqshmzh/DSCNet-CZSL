import argparse
import os.path

parser = argparse.ArgumentParser()

YML_PATH = {
    "general": "./config/general.yml",
    "mit-states": "./config/mit-states.yml",
    "ut-zap50k": "./config/ut-zap50k.yml",
    "cgqa": "./config/cgqa.yml",
    "clothing16k": "./config/clothing16k.yml",
    "ao-clevr": "./config/ao-clevr.yml",
    "vaw-czsl": "./config/vaw-czsl.yml",
}
# Params
parser.add_argument("--main_root", help="main root path of the code package", type=str, default=os.path.dirname(__file__))
parser.add_argument("--rank", help="rank of process", type=int)
parser.add_argument("--ddp", help="use distributed data parallel to train", default=False, type=bool)
parser.add_argument("--port", help="port of ddp", type=str, default="12345")
parser.add_argument("--device_ids", help="identity number of gpu", type=str, default="0")
parser.add_argument("--lr", help="learning rate", type=float, default=5e-05)
parser.add_argument("--dataset", help="name of the dataset", type=str, default="mit-states")
parser.add_argument("--weight_decay", help="weight decay", type=float, default=1e-05)
parser.add_argument("--pin_memory", action="store_true")
parser.add_argument("--persistent_workers", action="store_true")
parser.add_argument("--use_amp", help="use Auto Mixture Precision for training", action="store_true")
parser.add_argument("--epochs", help="number of epochs", default=20, type=int)
parser.add_argument("--epoch_start", help="start epoch", type=int)
parser.add_argument("--train_batch_size", help="train batch size", type=int)
parser.add_argument("--num_workers", default=8, type=int)
parser.add_argument("--eval_batch_size", help="eval batch size", type=int)
parser.add_argument("--save_path", help="save path", type=str)
parser.add_argument("--save_every_n", default=1, type=int, help="saves the model every n epochs")
parser.add_argument("--save_model", type=bool, default=True, help="indicate if you want to save the model")
parser.add_argument("--model_load", default=None, type=str, help="file name of the loaded model")
parser.add_argument("--seed", help="seed value", default=0, type=int)
parser.add_argument("--gradient_accumulation_steps", help="number of gradient accumulation steps", default=1, type=float)
parser.add_argument("--device", type=str)
parser.add_argument("--open_world", help="evaluate on open world setup", action="store_true")
parser.add_argument("--threshold", type=float, help="optional threshold")
parser.add_argument("--bias", help="eval bias", type=float, default=1e3)
parser.add_argument("--topk", help="eval topk", type=int, default=1)

''' Model Structure Params '''
parser.add_argument("--visual_extractor", help="visual feature extractor: clip | resnet18", type=str, default="clip")
parser.add_argument("--pooling_mode", type=str, default="mean", help="Choose from cls|sum|mean|clip")
parser.add_argument("--custom_post_proj", type=bool, default=False, help="Custom post layernorm and projection if clip proj is not used")
parser.add_argument("--context_length", help="sets the context length of the clip model", default=8, type=int)
parser.add_argument("--prompt_dropout", help="add dropout to prompt", type=float, default=0.3)
parser.add_argument("--visual_emb_mapper", type=str, default="self-attention", help="Mapper type for visual features: mlp | self-attention")
parser.add_argument("--cosine_scale", type=float, help='temperature of cosine similarity in CE loss')
parser.add_argument("--alpha_2", type=float)
parser.add_argument("--alpha_3", type=float)
parser.add_argument("--feat_dim", type=float, default=768)
# Visual extractor related parameters
parser.add_argument("--clip_model", help="clip model type", type=str, default="ViT-L/14")
parser.add_argument("--clip_path", help="path to clip model file", type=str)