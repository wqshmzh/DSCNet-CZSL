import clip
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.clip_modules.model_loader import load
from model.common import *


class DSCNet(nn.Module):
    def __init__(self, config, attributes, objects, offset):
        super().__init__()
        feat_dim = config.feat_dim
        vision_dim = 1024
        self.clip, _ = load(config.clip_model, context_length=config.context_length, download_root=config.clip_path)
        for p in self.clip.parameters():
            p.requires_grad=False
        self.context_length = config.context_length
        self.config = config
        self.objects = objects
        self.attributes = attributes
        self.n_attr = len(attributes)
        self.n_obj = len(objects)
        self.offset = offset
        self.prompt_token_ids_pair, self.prompt_token_ids_attr, self.prompt_token_ids_obj, soft_att_obj, \
            ctx_vectors, ctx_vectors_attr = self.construct_soft_prompt()
        self.attr_eos_id = int(self.prompt_token_ids_attr[0].argmax())
        self.obj_eos_id = int(self.prompt_token_ids_obj[0].argmax())
        self.pair_eos_id = int(self.prompt_token_ids_pair[0].argmax())
        self.enable_pos_emb = True
        
        # text prompt
        self.soft_att_obj = nn.Parameter(soft_att_obj)
        self.context_prompt_attr = nn.Parameter(ctx_vectors)
        self.context_prompt_obj = nn.Parameter(ctx_vectors.clone())
        self.context_prompt_pair = nn.Parameter(ctx_vectors.clone())
        self.object_prompt_attr = nn.Parameter(ctx_vectors_attr)
        
        self.attr_obj_dropout = nn.Dropout(config.prompt_dropout)
        self.attr_prefix_dropout = nn.Dropout(config.prompt_dropout)
        self.attr_postfix_dropout = nn.Dropout(config.prompt_dropout)
        self.obj_prefix_dropout = nn.Dropout(config.prompt_dropout)
        self.pair_prefix_dropout = nn.Dropout(config.prompt_dropout)
        
        num_heads_vis = vision_dim // 64
        num_heads = feat_dim // 64
        
        self.attr_learner = CrossResidualAttentionBlock(feat_dim, num_heads, config.attn_dropout, config.mlp_dropout)
        self.obj_learner = CrossResidualAttentionBlock(feat_dim, num_heads, config.attn_dropout, config.mlp_dropout)
        
        self.attr_vision_learner = CrossResidualAttentionBlock(feat_dim, num_heads, config.attn_dropout, config.mlp_dropout)
        self.obj_vision_learner = CrossResidualAttentionBlock(feat_dim, num_heads, config.attn_dropout, config.mlp_dropout)

        '''======================= Visual mapper ======================='''
        self.image_embedder_attr = SelfResidualAttentionBlock(vision_dim, num_heads_vis, config.attn_dropout, config.mlp_dropout)
        self.image_embedder_obj = SelfResidualAttentionBlock(vision_dim, num_heads_vis, config.attn_dropout, config.mlp_dropout)
        self.image_embedder_pair = SelfResidualAttentionBlock(vision_dim, num_heads_vis, config.attn_dropout, config.mlp_dropout)

        '''======================================================'''
        # Convert 1024 to 768 dims
        self.img_feats_down = nn.Sequential(nn.Linear(vision_dim, feat_dim), nn.LayerNorm(feat_dim), nn.Dropout(config.proj_dropout))
        self.attr_img_proj = nn.Sequential(nn.Linear(feat_dim, feat_dim), nn.LayerNorm(feat_dim), nn.Dropout(config.proj_dropout))
        self.attr_text_proj = nn.Sequential(nn.Linear(feat_dim, feat_dim), nn.LayerNorm(feat_dim), nn.Dropout(config.proj_dropout))
        self.obj_img_proj = nn.Sequential(nn.Linear(feat_dim, feat_dim), nn.LayerNorm(feat_dim), nn.Dropout(config.proj_dropout))
        self.obj_text_proj = nn.Sequential(nn.Linear(feat_dim, feat_dim), nn.LayerNorm(feat_dim), nn.Dropout(config.proj_dropout))
        
        self.attr_condition_learner = SelfResidualAttentionBlock(feat_dim, num_heads, config.attn_dropout, config.mlp_dropout)
        self.obj_condition_learner = SelfResidualAttentionBlock(feat_dim, num_heads, config.attn_dropout, config.mlp_dropout)

        self.cosine_scale = self.clip.logit_scale.exp()
        
        self.alpha_1 = config.alpha_1
        self.alpha_2 = config.alpha_2
        self.alpha_3 = config.alpha_3
        
    
    def mapping_visual_features(self, mapper, vis_features, only_emb=False, **kwargs):
        vis_feats_mapped = mapper(vis_features) # 1024
        vis_embs_mapped = self.clip.visual.ln_post(vis_feats_mapped[:, 0])
        if self.clip.visual.proj is not None:
            vis_embs_mapped = vis_embs_mapped.type(self.clip.dtype) @ self.clip.visual.proj
            vis_embs_mapped = vis_embs_mapped.float()
        if not only_emb:
            vis_feats_mapped = kwargs['proj'](vis_feats_mapped) # 1024 -> 768
        else:
            vis_feats_mapped = None
        return vis_feats_mapped, vis_embs_mapped
    
    def text_token_pooling(self, proj, text_token_feats, eos_idx):
        text_embs = proj(text_token_feats[:, eos_idx])
        return text_embs

    def construct_soft_prompt(self):
        prompt_token_ids_pair = clip.tokenize("a photo of x x", context_length=self.config.context_length).cuda()
        prompt_token_ids_attr = clip.tokenize("a photo of x object", context_length=self.config.context_length).cuda()
        prompt_token_ids_obj = clip.tokenize("a photo of x", context_length=self.config.context_length).cuda()
        tokenized = torch.cat(
            [
                clip.tokenize(tok, context_length=self.config.context_length)
                for tok in self.attributes + self.objects
            ]
        )
        with torch.no_grad():
            orig_token_embedding = self.clip.token_embedding(tokenized.cuda())

        soft_att_obj = torch.zeros(
            (len(self.attributes) + len(self.objects), orig_token_embedding.size(-1)),
        )
        for idx, rep in enumerate(orig_token_embedding):
            eos_idx = tokenized[idx].argmax()
            soft_att_obj[idx, :] = torch.mean(rep[1:eos_idx, :], axis=0)

        ctx_init = "a photo of"
        n_ctx = len(ctx_init.split())
        prompt = clip.tokenize(ctx_init, context_length=self.config.context_length).cuda()
        with torch.no_grad():
            embedding = self.clip.token_embedding(prompt)
        ctx_vectors = embedding[0, 1:1+n_ctx, :]
        
        ctx_init = "object"
        prompt = clip.tokenize(ctx_init,
                            context_length=self.config.context_length).cuda() # (1,8)
        with torch.no_grad():
            embedding = self.clip.token_embedding(prompt)
        ctx_vectors_attr = embedding[0, 1:2, :] # (1,feat_dim)

        return prompt_token_ids_pair, prompt_token_ids_attr, prompt_token_ids_obj, soft_att_obj, ctx_vectors, ctx_vectors_attr

    def construct_full_pair_prompts(self, pair_idx):
        attr_idx, obj_idx = pair_idx[:, 0], pair_idx[:, 1]
        prompt_token_ids_pair = self.prompt_token_ids_pair.repeat(len(pair_idx), 1)
        pair_full_prompts = self.clip.token_embedding(prompt_token_ids_pair)
        pair_full_prompts[:, self.pair_eos_id - 2, :] = self.attr_obj_dropout(self.soft_att_obj[attr_idx])
        pair_full_prompts[:, self.pair_eos_id - 1, :] = self.attr_obj_dropout(self.soft_att_obj[obj_idx+self.offset])

        # adding the correct learnable context
        pair_full_prompts[:, 1:len(self.context_prompt_pair)+1, :] = self.pair_prefix_dropout(self.context_prompt_pair)
        
        return pair_full_prompts

    def construct_full_attr_obj_prompts(self, pair_idx):
        attr_idx, obj_idx = pair_idx[:, 0], pair_idx[:, 1] # (1262,2)
        
        attr_set = torch.unique(attr_idx)
        obj_set = torch.unique(obj_idx)
        
        # Construct attr soft prompt=======================================
        prompt_token_ids_attr = self.prompt_token_ids_attr.repeat(len(attr_set), 1)
        attr_full_prompts = self.clip.token_embedding(prompt_token_ids_attr) # (115,8,feat_dim)
        attr_full_prompts[:, self.attr_eos_id-2, :] = self.attr_obj_dropout(self.soft_att_obj[:self.offset]) # attr 
        # adding the correct learnable context
        attr_full_prompts[:, 1:len(self.context_prompt_attr)+1, :] = self.attr_prefix_dropout(self.context_prompt_attr) # (3,feat_dim)
        attr_full_prompts[:, self.attr_eos_id-1, :] = self.attr_postfix_dropout(self.object_prompt_attr)

        # Construct obj soft prompt========================================
        prompt_token_ids_obj = self.prompt_token_ids_obj.repeat(len(obj_set), 1)
        obj_full_prompts = self.clip.token_embedding(prompt_token_ids_obj) # (245,8,feat_dim)
        obj_full_prompts[:, self.obj_eos_id-1, :] = self.attr_obj_dropout(self.soft_att_obj[self.offset:]) # obj
        # adding the correct learnable context
        obj_full_prompts[:, 1:len(self.context_prompt_obj)+1, :] = self.obj_prefix_dropout(self.context_prompt_obj) # (3.feat_dim)

        return attr_full_prompts, obj_full_prompts

    def encode_image(self, x: torch.Tensor):
        x = self.clip.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.clip.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.clip.visual.positional_embedding.to(x.dtype)
        x = self.clip.visual.ln_pre(x)
        
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip.visual.transformer(x)
        x_features = x.permute(1, 0, 2)  # LND -> NLD

        x = self.clip.visual.ln_post(x_features[:, 0])
        if self.clip.visual.proj is not None:
            x = x @ self.clip.visual.proj
        return x.float(), x_features.float()
    
    def encode_text(self, text_embeddings, eos_idx):
        text_embeddings = text_embeddings.type(self.clip.dtype)
        x = (
            text_embeddings + self.clip.positional_embedding.type(self.clip.dtype)
            if self.enable_pos_emb
            else text_embeddings
        )
        x = x.permute(1, 0, 2)
        x = self.clip.transformer(x)
        text_features = x.permute(1, 0, 2)
        x = self.clip.ln_final(text_features)
        tf = (
            x[
                torch.arange(x.shape[0]), eos_idx
            ]  # POS of <EOS>
            @ self.clip.text_projection
        )
        return tf.float(), text_features.float()
    
    def normalize_cosine(self, cos_logits):
        probability = (cos_logits + 1) / 2
        return probability

    def forward(self, batch, idx):
        if self.training:
            x, attr_gt, obj_gt, pair_gt = batch[0], batch[1], batch[2], batch[3]
        else:
            x = batch[0]
        batch_size = x.shape[0]
        
        # Compute visual features
        x, x_features = self.encode_image(x.type(self.clip.dtype))
        
        # Map visual features into different spaces
        img_feats_attr, img_proj_attr = self.mapping_visual_features(
            self.image_embedder_attr, vis_features=x_features, proj=self.img_feats_down
            )
        img_proj_pair = self.mapping_visual_features(
            self.image_embedder_pair, vis_features=x_features, only_emb=True
            )[1]
        img_feats_obj, img_proj_obj = self.mapping_visual_features(
            self.image_embedder_obj, vis_features=x_features, proj=self.img_feats_down
            ) # batch_size, tokens, channels
        
        x_features = self.img_feats_down(x_features)
        
        img_proj_attr = img_proj_attr + x
        img_proj_obj = img_proj_obj + x
        img_proj_pair = (img_proj_attr + img_proj_obj + img_proj_pair) / 3
        img_proj_pair = img_proj_pair + x
        
        # Normalize image embedding
        img_proj_attr = F.normalize(img_proj_attr, dim=-1)
        img_proj_pair = F.normalize(img_proj_pair, dim=-1)
        img_proj_obj = F.normalize(img_proj_obj, dim=-1)

        # Acquire word embeddings of all attrs and objs
        attr_full_prompts, obj_full_prompts = self.construct_full_attr_obj_prompts(idx)
        pair_full_prompts = self.construct_full_pair_prompts(idx)
        
        # Encode text prompts with CLIP transformer
        attr_prompt_embs, attr_prompt_feats = self.encode_text(attr_full_prompts, self.attr_eos_id)
        attr_prompt_embs = F.normalize(attr_prompt_embs, dim=-1) # n_attr, 1, channels
        
        obj_prompt_embs, obj_prompt_feats = self.encode_text(obj_full_prompts, self.obj_eos_id)
        obj_prompt_embs = F.normalize(obj_prompt_embs, dim=-1)
        
        pair_prompt_embs = self.encode_text(pair_full_prompts, self.pair_eos_id)[0]
        pair_prompt_embs = F.normalize(pair_prompt_embs, dim=-1)

        ''' Conditional Attributes Learning '''
        # Pred object
        pre_cosine_dist_obj = img_proj_obj @ obj_prompt_embs.t() # batch_size * n_obj
        pre_cosine_dist_obj_norm = self.normalize_cosine(pre_cosine_dist_obj)
        pred_obj = torch.argmax(pre_cosine_dist_obj_norm, dim=-1)
        pred_obj_prompt_feats = obj_prompt_feats[pred_obj]
        # Conditional attribute prompt learning
        attr_condition = torch.cat((pred_obj_prompt_feats, x_features), dim=1)
        encoded_attr_text_condition = self.attr_condition_learner(attr_condition) # batch_size, tokens, channel
        conditional_attr_embs = []
        for b in range(batch_size):
            conditional_attr_feats = self.attr_learner(attr_prompt_feats, encoded_attr_text_condition[b:b+1].repeat(self.n_attr, 1, 1))
            conditional_attr_embs.append(self.text_token_pooling(self.attr_text_proj, conditional_attr_feats, self.attr_eos_id).unsqueeze(dim=0))# n_attr, channels
        conditional_attr_embs = torch.cat(conditional_attr_embs)
        conditional_attr_embs = F.normalize(conditional_attr_embs, dim=-1)
        conditional_attr_img_embs = self.attr_img_proj(self.attr_vision_learner(img_feats_attr, pred_obj_prompt_feats)[:, 0])
        conditional_attr_img_embs = F.normalize(conditional_attr_img_embs, dim=-1)
        cosine_dist_attr = torch.einsum('bd,bad->ba', conditional_attr_img_embs, conditional_attr_embs)
        prob_attr = self.normalize_cosine(cosine_dist_attr)
        
        ''' Conditional Objects Learning '''
        # Pred attribute
        pre_cosine_dist_attr = img_proj_attr @ attr_prompt_embs.t()
        pre_cosine_dist_attr_norm = self.normalize_cosine(pre_cosine_dist_attr)
        pred_attr = torch.argmax(pre_cosine_dist_attr_norm, dim=-1)
        pred_attr_prompt_feats = attr_prompt_feats[pred_attr]
        # Conditional object prompt learning
        obj_condition = torch.cat((pred_attr_prompt_feats, x_features), dim=1)
        obj_text_condition = self.obj_condition_learner(obj_condition) # batch_size, tokens, channel
        conditional_obj_embs = []
        for b in range(batch_size):
            conditional_obj_feats = self.obj_learner(obj_prompt_feats, obj_text_condition[b:b+1].repeat(self.n_obj, 1, 1))
            conditional_obj_embs.append(self.text_token_pooling(self.obj_text_proj, conditional_obj_feats, self.obj_eos_id).unsqueeze(dim=0)) # n_attr, channels
        conditional_obj_embs = torch.cat(conditional_obj_embs)
        conditional_obj_embs = F.normalize(conditional_obj_embs, dim=-1)
        conditional_obj_img_embs = self.obj_img_proj(self.obj_vision_learner(img_feats_obj, pred_attr_prompt_feats)[:, 0])
        conditional_obj_img_embs = F.normalize(conditional_obj_img_embs, dim=-1)
        cosine_dist_obj = torch.einsum('bd,bod->bo', conditional_obj_img_embs, conditional_obj_embs)
        prob_obj = self.normalize_cosine(cosine_dist_obj)
        
        ''' Pred composition '''
        cosine_dist_pair = torch.einsum('bd,pd->bp', img_proj_pair, pair_prompt_embs)
        prob_pair = self.normalize_cosine(cosine_dist_pair)
        
        scores = torch.zeros((batch_size, cosine_dist_pair.shape[1])).to(self.config.device)
        for pair_idx, pair in enumerate(idx):
            attr_idx = pair[0]
            obj_idx = pair[1]
            scores[:, pair_idx] = (1 - self.alpha_1) * prob_pair[:, pair_idx] + \
                self.alpha_1 * 0.5 * (prob_attr[:, attr_idx] * pre_cosine_dist_obj_norm[:, obj_idx] + \
                    prob_obj[:, obj_idx] * pre_cosine_dist_attr_norm[:, attr_idx])
        
        if self.training:
            L_o = F.cross_entropy(cosine_dist_obj*self.cosine_scale, obj_gt)
            L_a = F.cross_entropy(cosine_dist_attr*self.cosine_scale, attr_gt)
            L_ao = F.cross_entropy(cosine_dist_pair*self.cosine_scale, pair_gt)

            L = F.cross_entropy(scores*self.cosine_scale, pair_gt)
            return self.alpha_2 * (L_a + L_o) + self.alpha_3 * L_ao + L
        else:
            return scores