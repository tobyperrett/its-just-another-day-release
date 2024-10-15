# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Part of the code is from https://github.com/huggingface/transformers/blob/main/src/transformers/generation_utils.py
# Modified by Yue Zhao
# The original code is under Apache 2.0 License


import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from transformers import BeamSearchScorer
from transformers.generation.logits_process import (
    LogitsProcessorList,
    TopKLogitsWarper,
    TopPLogitsWarper,
    TemperatureLogitsWarper,
    TypicalLogitsWarper,
    LogitNormalization,
)

from lavila.models.coca import CrossAttention, LayerNorm
from lavila.models.openai_model import VisionTransformer
from lavila.models.timesformer import SpaceTimeTransformer
from lavila.utils import distributed as dist_utils


import lavila.models.estimator as estimator



class UCT(nn.Module):
    def __init__(self,
                 vision_width: int,
                 max_videos=10,
                 args=None,
                 num_img_queries=256,
                 ):
        
        super().__init__()
        self.max_videos = max_videos
        self.rand_weights = nn.Parameter(torch.ones((self.max_videos)), requires_grad=False )
        self.args = args

        dim_feedforward = self.args.uct_dim_feedforward
        n_encoder_layers = self.args.uct_layers
        n_head = self.args.uct_heads




        self.video_embedding = nn.Parameter(torch.empty(self.max_videos, vision_width))
        if self.args.img_q_pe:
            self.img_q_pe = nn.Parameter(torch.empty(num_img_queries, vision_width))


        if self.args.uct_type == "attention":
            self.transformer_encoder = nn.MultiheadAttention(vision_width, n_head, batch_first=True)
        elif self.args.uct_type == 'encoder':
            encoder_layer = nn.TransformerEncoderLayer(d_model=vision_width, nhead=n_head, dim_feedforward=dim_feedforward, batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_encoder_layers)
        elif self.args.uct_type == 'weighted':
            encoder_layer = nn.TransformerEncoderLayer(d_model=vision_width, nhead=n_head, dim_feedforward=dim_feedforward, batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_encoder_layers)
            self.token_weights = nn.Linear(vision_width, 1)
        elif self.args.uct_type == 'residual':
            encoder_layer = nn.TransformerEncoderLayer(d_model=vision_width, nhead=n_head, dim_feedforward=dim_feedforward, batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_encoder_layers)
        elif self.args.uct_type == 'residual2':
            encoder_layer = nn.TransformerEncoderLayer(d_model=vision_width, nhead=n_head, dim_feedforward=dim_feedforward, batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_encoder_layers)
            self.scale = nn.Parameter(torch.empty(1))
            self.layer_norm = LayerNorm(vision_width)

        elif self.args.uct_type == 'shared-residual':
            encoder_layer = nn.TransformerEncoderLayer(d_model=vision_width, nhead=n_head, dim_feedforward=dim_feedforward, batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_encoder_layers)
        elif self.args.uct_type == 'none-debug':
            pass
        else:
            raise ValueError("Invalid UCT architecture")
    

    def initialize_parameters(self):
        # nn.init.normal_(self.video_embedding, std=self.vision_width ** -0.5)
        nn.init.normal_(self.video_embedding, std=0.01)
        if self.args.img_q_pe:
            nn.init.normal_(self.img_q_pe, std=0.01)

        if self.args.uct_type == 'none-debug':
            pass
        else:
            for name, param in self.transformer_encoder.named_parameters():
                if 'weight' in name and param.data.dim() == 2:
                    nn.init.xavier_uniform_(param)  
        
        if self.args.uct_type == "weighted":
            nn.init.normal_(self.token_weights.weight, std=0.02)
        if self.args.uct_type == "residual2":
            nn.init.constant_(self.scale, 0.0)


    def forward(self, x):
        bb, mm, tt, dd = x.shape

        if self.args.img_q_pe:
            x = x + self.img_q_pe

        x = rearrange(x, 'b m t d -> b t m d')
        
        # select random video embeddings - emsures first video is not always tagged the same
        batch_vid_embed_idxs = torch.multinomial(self.rand_weights, mm, replacement=False)
        batch_vid_embeds = torch.index_select(self.video_embedding, 0, batch_vid_embed_idxs)

        if self.args.uct_type == 'none-debug':
            batch_vid_embed_idxs = torch.tanh(batch_vid_embed_idxs) * 0

        x = x + batch_vid_embeds
        x = rearrange(x, 'b t m d -> b (m t) d')

        if self.args.uct_type == "attention":
            x = self.attn(x, x, x)[0]
        elif self.args.uct_type == 'encoder':
            x = self.transformer_encoder(x)
        elif self.args.uct_type == 'weighted':
            r = self.transformer_encoder(x)
            r = self.token_weights(r)
            r = torch.tanh(r)
            x = x * r
        elif self.args.uct_type == 'residual':
            r = self.transformer_encoder(x)
            # r = torch.tanh(r)
            x = x - r
        elif self.args.uct_type == 'residual2':
            s = torch.nn.functional.tanh(self.scale)
            r = self.transformer_encoder(x)
            r = self.layer_norm(r)
            x = x - r * s
        elif self.args.uct_type == 'shared-residual':
            r = self.transformer_encoder(x)

        # needs to be done on everything
        x = rearrange(x, 'b (m t) d -> b m t d', m=mm, t=tt)
        
        if self.args.uct_type == 'shared-residual':
            r = torch.mean(r, dim=-3, keepdim=True)
            x = x - r

        return x



class VCLM_HF(nn.Module):
    def __init__(self,
                 # vision
                 vision_width: int,
                 vision_model: nn.Module,
                 # text
                 text_width: int,
                 text_decoder: nn.Module,
                 num_img_queries=256,
                 dim_head=64,
                 heads=8,
                 **kwargs,
                 ):
        super().__init__()

        self.vision_width = vision_width
        self.visual = vision_model
        self.text_width = text_width
        self.text_decoder = text_decoder
        self.tokenizer = kwargs["tokenizer"]
        self.args = kwargs["args"]

        # override num_img_queries with version in args
        num_img_queries = self.args.n_img_q

        if self.args.estimator is not None:
            self.estimator = getattr(estimator, self.args.estimator)(self.args)
        else:
            self.estimator = None

        self.img_queries = nn.Parameter(torch.empty(num_img_queries, text_width))
        self.img_attn_pool = CrossAttention(
            dim=text_width, context_dim=vision_width,
            dim_head=dim_head, heads=heads,
            norm_context=True
        )
        self.img_attn_pool_norm = LayerNorm(text_width)

        if self.args.uct_type is not None:
            self.uct = UCT(vision_width=vision_width, args=self.args, num_img_queries=num_img_queries)
            self.bos_token = nn.Parameter(torch.LongTensor([[self.tokenizer.bos_token_id]]), requires_grad=False)


        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.img_queries, std=self.text_width ** -0.5)
        if self.args.uct_type is not None:
           self.uct.initialize_parameters()

    def encode_image(self, image, use_checkpoint=False, cls_at_last=False):

        if isinstance(self.visual, VisionTransformer):
            # openai_model.VisionTransformer accepts (N, C, H, W) instead of (N, C, T, H, W)
            image = image.permute(0, 2, 1, 3, 4)  # BCTHW -> BTCHW
            bb, tt, _, _, _ = image.shape
            x = self.visual(image.reshape(-1, *image.shape[2:]), use_checkpoint=use_checkpoint, cls_at_last=False)  # NLD
            x = x.view(bb, tt, *x.shape[1:])
            x = x.permute(0, 3, 1, 2)
        elif isinstance(self.visual, SpaceTimeTransformer):

            shape = image.shape
            if len(shape) == 6:
                bb, nc, cc, tt, hh, ww = shape
                image = rearrange(image, 'b nc ... -> (b nc) ...')
                multiple_clips = True
            else:
                multiple_clips = False

            image = image.permute(0, 2, 1, 3, 4).contiguous()  # BCTHW -> BTCHW

            if self.args.estimator == "MLP":
                x, x_cls = self.visual.forward_features(image, use_checkpoint=use_checkpoint, cls_at_last=True, return_x_and_cls=True)  # NLD
            else:
            # elif (self.args.estimator is None) or (self.args.estimator == "MLP_2LN") or (self.args.estimator == 'TP_MLP2') or (self.args.estimator == 'TP_T1'):
                x = self.visual.forward_features(image, use_checkpoint=use_checkpoint, cls_at_last=cls_at_last)  # NLD

            # else:
            #     raise NotImplementedError(f"forward for estimator {self.args.estimator} not implemented")

            x = x.permute(0, 2, 1)
        else:
            x = self.visual(image, use_checkpoint=use_checkpoint, mean_at_last=False)
        if isinstance(x, list):
            assert len(x) == 1
            x = x[0]

        x = x.flatten(start_dim=2)  # BDTHW -> BD(THW)

        x = x.permute(0, 2, 1)      # BDN -> BND


        img_queries = repeat(self.img_queries, 'n d -> b n d', b=x.shape[0])
        img_queries = self.img_attn_pool(img_queries, x)
        img_queries = self.img_attn_pool_norm(img_queries)


        if multiple_clips:
            img_queries = rearrange(img_queries, '(b nc) cells ... -> b nc cells ...', b=bb, nc=nc)


        if self.args.estimator == "MLP":
            x_cls = rearrange(x_cls, '(b nc) cells ... -> b nc cells ...', b=bb, nc=nc)
            # print(img_queries.shape, x_cls.shape)
            return img_queries, x_cls

        return img_queries, None
    
    def get_sim_pred(self, vis, bos, bb, mm, combined=True):
        if combined:
            vis = rearrange(vis, '(b m) t d -> b m t d', b=bb, m=mm)
            bos = rearrange(bos, '(b m) d -> b m d', b=bb, m=mm)
            vis_r = torch.repeat_interleave(vis, vis.shape[-3], -3)
            bos_r = bos.repeat(1, bos.shape[-2], 1)
            vis_r = rearrange(vis_r, 'b mm t d -> (b mm) t d')
            bos_r = rearrange(bos_r, 'b mm d -> (b mm) d')
            estimator_output = self.estimator({'all_vis_feats': vis_r, 'all_bos_feats': bos_r})["output"]
            cos_sims = rearrange(estimator_output, '(b m1 m2) -> b m1 m2', b=bb, m1=mm, m2=mm)
            labels = torch.eye(mm, requires_grad=False).to(estimator_output.device)
            labels = labels.repeat(bb, 1)
        else:
            # print(vis.shape, bos.shape)
            est_output = self.estimator({'vis_feats': vis, 'bos_feats': bos})

            # print(est_output['vis_pred'].shape, est_output['text_pred'].shape)
            vis_pred = rearrange(est_output['vis_pred'], '(b m) d -> b m d', b=bb, m=mm)
            text_pred = rearrange(est_output['text_pred'], '(b m) d-> b m d', b=bb, m=mm)
            # print(vis_pred.shape, text_pred.shape)
            vis_pred_r = torch.repeat_interleave(vis_pred, vis_pred.shape[-2], -2)
            text_pred_r = text_pred.repeat(1, text_pred.shape[-2], 1)
            # print(vis_pred_r.shape, text_pred_r.shape)
            cos_sims = torch.cosine_similarity(vis_pred_r, text_pred_r, dim=-1)
            # print(cos_sims.shape)
            cos_sims = rearrange(cos_sims, 'b (m1 m2) -> b m1 m2', m1 = mm, m2 = mm)
            labels = torch.eye(mm, requires_grad=False).to(cos_sims.device)
            labels = labels.repeat(bb, 1)
            # print(cos_sims.shape, labels.shape)
            # exit(0)

        return cos_sims, labels
        

    def forward(self, image, text, mask=None, use_checkpoint=False, norm_embed=False, return_features=False):
        log_dict = {}
        
        if use_checkpoint:
            self.text_decoder.gradient_checkpointing_enable()
        else:
            self.text_decoder.gradient_checkpointing_disable()

        image_tokens, x_cls = self.encode_image(image, use_checkpoint=use_checkpoint)

        return_dict = {}

        if len(image_tokens.shape) == 4:
            bb, mm, tt, dd = image_tokens.shape

            image_tokens_u = self.uct(image_tokens)
            image_tokens_u = self.img_attn_pool_norm(image_tokens_u)
            u_vid_difference = 1 - F.cosine_similarity(image_tokens_u, image_tokens, dim=-1).mean(dim=-1)
            log_dict["u vid difference"] = u_vid_difference.mean()           

            # if self.args.estimator == "MLP":
            #     # reshape to run through decoder
            #     image_tokens_u = rearrange(image_tokens_u, 'b m t d -> (b m) t d')
            #     condition_text_ids = self.bos_token.repeat(bb*mm, 1)
            #     output_decoder = self.text_decoder(condition_text_ids, encoder_hidden_states=image_tokens_u, output_hidden_states=True)
            #     hidden = output_decoder.hidden_states.squeeze()
            #     hidden = rearrange(hidden, '(b m) d -> b m d', b=bb, m=mm)

            #     # cat mlp output with hidden states
            #     # compare every mlp output with every hidden state, for contrastive loss later on
            #     x_cls_r = torch.repeat_interleave(x_cls, x_cls.shape[-2], -2)
            #     hidden_r = hidden.repeat(1, hidden.shape[-2], 1)
            #     estimator_input = torch.cat([x_cls_r, hidden_r], dim=-1)
            #     estimator_output = self.estimator(estimator_input)
            #     estimator_output = rearrange(estimator_output, 'b (c h) ... -> b c h ...', c=hidden.shape[-2])
            #     estimator_output = estimator_output.squeeze(-1)
            #     labels = torch.eye(mm, requires_grad=False).to(estimator_output.device)
            #     labels = labels.repeat(bb, 1)

            #     # check hidden against cls of originals in the batch.
            #     # should still be able to contrastively tell them apart.
            #     # for now, just take one video of each similarity tuple.
            #     x_cls_singles = x_cls[:, 0]
            #     x_cls_singles_r = torch.repeat_interleave(x_cls_singles, x_cls_singles.shape[0], 0)
            #     hidden_singles = hidden[:, 0]
            #     hidden_singles_r = hidden_singles.repeat(hidden_singles.shape[0], 1)
            #     estimator_input_s = torch.cat([x_cls_singles_r, hidden_singles_r], dim=-1)
            #     estimator_output_s = self.estimator(estimator_input_s).squeeze(-1)
            #     estimator_output_s = rearrange(estimator_output_s, '(b1 b2) -> b1 b2', b1=bb, b2=bb).unsqueeze(0)
            #     labels_s = torch.eye(bb, requires_grad=False).to(estimator_output_s.device)

            #     # # check hidden of originals against cls of u in the batch
            #     # x_cls_singles_t = x_cls[:, 0]
            #     # x_cls_singles_t_r = torch.repeat_interleave(x_cls_singles_t, x_cls_singles_t.shape[0], 0)
            #     # hidden_singles_t = hidden[:, 0]
            #     # hidden_singles_t_r = hidden_singles_t.repeat(hidden_singles_t.shape[0], 1)
            #     # estimator_input_t = torch.cat([x_cls_singles_t_r, hidden_singles_t_r], dim=-1)
            #     # estimator_output_t = self.estimator(estimator_input_t).squeeze(-1)
            #     # estimator_output_t = rearrange(estimator_output_t, '(b1 b2) -> b1 b2', b1=bb, b2=bb).unsqueeze(0)
            #     # labels_t = torch.eye(bb, requires_grad=False).to(estimator_output_t.device)

            #     # check original hidden and cls
            #     image_tokens = rearrange(image_tokens, 'b m t d -> (b m) t d')
            #     output_decoder_t = self.text_decoder(condition_text_ids, encoder_hidden_states=image_tokens, output_hidden_states=True)
            #     hidden_t = output_decoder_t.hidden_states.squeeze()
            #     hidden_t = rearrange(hidden_t, '(b m) d -> b m d', b=bb, m=mm)
            #     hidden_singles_t = hidden_t[:, 0]
            #     hidden_singles_t_r = hidden_singles_t.repeat(hidden_singles_t.shape[0], 1)
            #     estimator_input_b = torch.cat([x_cls_singles_r, hidden_singles_t_r], dim=-1)
            #     estimator_output_b = self.estimator(estimator_input_b).squeeze(-1)
            #     estimator_output_b = rearrange(estimator_output_b, '(b1 b2) -> b1 b2', b1=bb, b2=bb).unsqueeze(0)
            #     labels_b = torch.eye(bb, requires_grad=False).to(estimator_output_b.device)

            #     return_dict = {'sim_preds': estimator_output, 'labels': labels, 'sim_preds_s': estimator_output_s, 'labels_s': labels_s,  'sim_preds_b': estimator_output_b, 'labels_b': labels_b,  'u_vid_difference': u_vid_difference, 'log_dict': log_dict}
            #     return return_dict
            # elif self.args.estimator == 'TP_MLP2_old':
                # condition_text_ids = self.bos_token.repeat(bb*mm, 1)
                # device = image_tokens.device

                # o_vis = image_tokens
                # u_vis = image_tokens_u

                # u_vis = rearrange(u_vis, 'b m t d -> (b m) t d')
                # o_vis = rearrange(o_vis, 'b m t d -> (b m) t d')



                # # u vis to u text, sets
                # u_output_decoder = self.text_decoder(condition_text_ids, encoder_hidden_states=u_vis, output_hidden_states=True)
                # u_hidden = u_output_decoder.hidden_states.squeeze()
                # u_est_output = self.estimator({'vis_feats': u_vis, 'bos_feats': u_hidden})
                # u_vis_pred = rearrange(u_est_output['vis_pred'], '(b m) d -> b m d', b=bb, m=mm)
                # u_text_pred = rearrange(u_est_output['text_pred'], '(b m) d-> b m d', b=bb, m=mm)
                # u_vis_pred_r = torch.repeat_interleave(u_vis_pred, u_vis_pred.shape[-2], -2)
                # u_text_pred_r = u_text_pred.repeat(1, u_text_pred.shape[-2], 1)
                # u_u_cos_sims = torch.cosine_similarity(u_vis_pred_r, u_text_pred_r, dim=-1)
                # u_u_cos_sims = rearrange(u_u_cos_sims, 'b (m1 m2) -> b m1 m2', m1 = mm, m2 = mm)
                # u_u_labels = torch.eye(mm, requires_grad=False).to(device)
                # u_u_labels = u_u_labels.repeat(bb, 1)
                # return_dict['sim_preds'] = u_u_cos_sims
                # return_dict['labels'] = u_u_labels



                # # o vis to o text, sets
                # o_output_decoder = self.text_decoder(condition_text_ids, encoder_hidden_states=o_vis, output_hidden_states=True)
                # o_hidden = o_output_decoder.hidden_states.squeeze()
                # o_est_output = self.estimator({'vis_feats': o_vis, 'bos_feats': o_hidden})
                # o_vis_pred = rearrange(o_est_output['vis_pred'], '(b m) d -> b m d', b=bb, m=mm)
                # o_text_pred = rearrange(o_est_output['text_pred'], '(b m) d-> b m d', b=bb, m=mm)
                # o_vis_pred_r = torch.repeat_interleave(o_vis_pred, o_vis_pred.shape[-2], -2)
                # o_text_pred_r = o_text_pred.repeat(1, o_text_pred.shape[-2], 1)
                # o_o_cos_sims = torch.cosine_similarity(o_vis_pred_r, o_text_pred_r, dim=-1)
                # o_o_cos_sims = rearrange(o_o_cos_sims, 'b (m1 m2) -> b m1 m2', m1 = mm, m2 = mm)
                # o_o_labels = torch.eye(mm, requires_grad=False).to(device)
                # o_o_labels = o_o_labels.repeat(bb, 1)
                # return_dict['sim_preds_b'] = o_o_cos_sims
                # return_dict['labels_b'] = o_o_labels

                # #u vis to o text, batch

                # return return_dict







                # exit(0)




                # return_dict = {'sim_preds': estimator_output, 'labels': labels, 'sim_preds_s': estimator_output_s, 'labels_s': labels_s, 'sim_preds_t': estimator_output_t, 'labels_t': labels_t, 'sim_preds_b': estimator_output_b, 'labels_b': labels_b, 'sim_preds_bu': estimator_output_bu, 'labels_bu': labels_bu, 'u_vid_difference': u_vid_difference, 'log_dict': log_dict}
            # elif (self.args.estimator == "TP_T1") or (self.args.estimator == "TP_T3"):
            
            if self.args.estimator is not None:
            
                condition_text_ids = self.bos_token.repeat(bb*mm, 1)
                device = image_tokens.device
                o_vis = image_tokens
                u_vis = image_tokens_u
                u_vis = rearrange(u_vis, 'b m t d -> (b m) t d')
                o_vis = rearrange(o_vis, 'b m t d -> (b m) t d')

                # u vis to u text, sets
                u_output_decoder = self.text_decoder(condition_text_ids, encoder_hidden_states=u_vis, output_hidden_states=True)
                u_hidden = u_output_decoder.hidden_states.squeeze()                

                o_output_decoder = self.text_decoder(condition_text_ids, encoder_hidden_states=o_vis, output_hidden_states=True)
                o_hidden = o_output_decoder.hidden_states.squeeze()

                return_dict['u_u_sim_preds'], return_dict['u_u_labels'] = self.get_sim_pred(u_vis, u_hidden, bb, mm, combined=self.estimator.combined)
                return_dict['o_o_sim_preds'], return_dict['o_o_labels'] = self.get_sim_pred(o_vis, o_hidden, bb, mm, combined=self.estimator.combined)
                return_dict['u_o_sim_preds'], return_dict['u_o_labels'] = self.get_sim_pred(u_vis, o_hidden, bb, mm, combined=self.estimator.combined)
                return_dict['o_u_sim_preds'], return_dict['o_u_labels'] = self.get_sim_pred(o_vis, u_hidden, bb, mm, combined=self.estimator.combined)

                b_u_vis = rearrange(u_vis, '(b m) t d -> b m t d', b=bb, m=mm )[:, 0]
                b_u_hidden = rearrange(u_hidden, '(b m) d -> b m d', b=bb, m=mm )[:, 0]
                b_o_vis = rearrange(o_vis, '(b m) t d -> b m t d', b=bb, m=mm )[:, 0]
                b_o_hidden = rearrange(o_hidden, '(b m) d -> b m d', b=bb, m=mm )[:, 0]

                return_dict['b_u_u_sim_preds'], return_dict['b_u_u_labels'] = self.get_sim_pred(b_u_vis, b_u_hidden, 1, bb, combined=self.estimator.combined)
                return_dict['b_o_o_sim_preds'], return_dict['b_o_o_labels'] = self.get_sim_pred(b_o_vis, b_o_hidden, 1, bb, combined=self.estimator.combined)
                return_dict['b_u_o_sim_preds'], return_dict['b_u_o_labels'] = self.get_sim_pred(b_u_vis, b_o_hidden, 1, bb, combined=self.estimator.combined)
                return_dict['b_o_u_sim_preds'], return_dict['b_o_u_labels'] = self.get_sim_pred(b_o_vis, b_u_hidden, 1, bb, combined=self.estimator.combined)

                return return_dict


            # elif (self.args.estimator == "MLP_2LN"):

            #     x_cls = image_tokens.mean(dim=-2)
            #     x_cls_u = image_tokens_u.mean(dim=-2)

            #     image_tokens_u = rearrange(image_tokens_u, 'b m t d -> (b m) t d')
            #     condition_text_ids = self.bos_token.repeat(bb*mm, 1)
            #     output_decoder = self.text_decoder(condition_text_ids, encoder_hidden_states=image_tokens_u, output_hidden_states=True)
            #     hidden = output_decoder.hidden_states.squeeze()
            #     hidden = rearrange(hidden, '(b m) d -> b m d', b=bb, m=mm)
                
            #     # cat mlp output with hidden states
            #     # compare every mlp output with every hidden state, for contrastive loss later on
            #     x_cls_r = torch.repeat_interleave(x_cls_u, x_cls_u.shape[-2], -2)
            #     hidden_r = hidden.repeat(1, hidden.shape[-2], 1)

            #     # estimator_input = torch.cat([x_cls_r, hidden_r], dim=-1)
            #     # estimator_output = self.estimator(estimator_input)
            #     # data_dict = {'vis_feats': x_cls_r, 'bos_feats': hidden_r}
            #     # estimator_output = self.estimator(data_dict)
            #     estimator_output = self.estimator(x_cls_r, hidden_r)
            #     estimator_output = rearrange(estimator_output, 'b (c h) ... -> b c h ...', c=hidden.shape[-2])
            #     estimator_output = estimator_output.squeeze(-1)
            #     labels = torch.eye(mm, requires_grad=False).to(estimator_output.device)
            #     labels = labels.repeat(bb, 1)

            #     # check hidden against cls of originals in the batch.
            #     # should still be able to contrastively tell them apart.
            #     # for now, just take one video of each similarity tuple.
            #     x_cls_singles = x_cls[:, 0]
            #     x_cls_singles_r = torch.repeat_interleave(x_cls_singles, x_cls_singles.shape[0], 0)
            #     hidden_singles = hidden[:, 0]
            #     hidden_singles_r = hidden_singles.repeat(hidden_singles.shape[0], 1)
            #     # estimator_input_s = torch.cat([x_cls_singles_r, hidden_singles_r], dim=-1)
            #     estimator_output_s = self.estimator(x_cls_singles_r, hidden_singles_r).squeeze(-1)
            #     # data_dict = {'vis_feats': x_cls_singles_r, 'bos_feats': hidden_singles_r}
            #     # estimator_output_s = self.estimator(data_dict)

            #     estimator_output_s = rearrange(estimator_output_s, '(b1 b2) -> b1 b2', b1=bb, b2=bb).unsqueeze(0)
            #     labels_s = torch.eye(bb, requires_grad=False).to(estimator_output_s.device)

            #     # # check hidden of originals against cls of u in the batch
            #     image_tokens = rearrange(image_tokens, 'b m t d -> (b m) t d')
            #     output_decoder_t = self.text_decoder(condition_text_ids, encoder_hidden_states=image_tokens, output_hidden_states=True)
            #     hidden_t = output_decoder_t.hidden_states.squeeze()
            #     hidden_t = rearrange(hidden_t, '(b m) d -> b m d', b=bb, m=mm)
            #     x_cls_singles_t = x_cls_u[:, 0]
            #     x_cls_singles_t_r = torch.repeat_interleave(x_cls_singles_t, x_cls_singles_t.shape[0], 0)
            #     hidden_singles_t = hidden_t[:, 0]
            #     hidden_singles_t_r = hidden_singles_t.repeat(hidden_singles_t.shape[0], 1)
            #     estimator_output_t = self.estimator(x_cls_singles_t_r, hidden_singles_t_r).squeeze(-1)

            #     # data_dict = {'vis_feats': x_cls_singles_t_r, 'bos_feats': hidden_singles_t_r}
            #     # estimator_output_t = self.estimator(data_dict)

            #     estimator_output_t = rearrange(estimator_output_t, '(b1 b2) -> b1 b2', b1=bb, b2=bb).unsqueeze(0)
            #     labels_t = torch.eye(bb, requires_grad=False).to(estimator_output_t.device)

            #     # check original hidden and cls
            #     estimator_output_b = self.estimator(x_cls_singles_r, hidden_singles_t_r).squeeze(-1)
            #     estimator_output_b = rearrange(estimator_output_b, '(b1 b2) -> b1 b2', b1=bb, b2=bb).unsqueeze(0)
            #     labels_b = torch.eye(bb, requires_grad=False).to(estimator_output_b.device)

            #     # check u hidden and cls
            #     estimator_output_bu = self.estimator(x_cls_singles_t_r, hidden_singles_r).squeeze(-1)
            #     # data_dict = {'vis_feats': x_cls_singles_t_r, 'bos_feats': hidden_singles_r}
            #     # estimator_output_bu = self.estimator(data_dict)
            #     estimator_output_bu = rearrange(estimator_output_bu, '(b1 b2) -> b1 b2', b1=bb, b2=bb).unsqueeze(0)
            #     labels_bu = torch.eye(bb, requires_grad=False).to(estimator_output_bu.device)

            #     return_dict = {'sim_preds': estimator_output, 'labels': labels, 'sim_preds_s': estimator_output_s, 'labels_s': labels_s, 'sim_preds_t': estimator_output_t, 'labels_t': labels_t, 'sim_preds_b': estimator_output_b, 'labels_b': labels_b, 'sim_preds_bu': estimator_output_bu, 'labels_bu': labels_bu, 'u_vid_difference': u_vid_difference, 'log_dict': log_dict}
                
            #     # return_dict = {'sim_preds': estimator_output, 'labels': labels, 'sim_preds_s': estimator_output_s, 'labels_s': labels_s, 'u_vid_difference': u_vid_difference, 'log_dict': log_dict}
            #     return return_dict
            # elif self.args.estimator is not None:
            #     raise NotImplementedError(f"forward for estimator {self.args.estimator} not implemented")
            else:
                # just do standard VCLM - need to reshape text as well
                # raise NotImplementedError(f"forward for estimator {self.args.estimator} not implemented")
                image_tokens_u = rearrange(image_tokens_u, 'b m t d -> (b m) t d')
                image_tokens_for_lm = image_tokens_u
                text = rearrange(text, 'b m ... -> (b m) ...')
        else:
            image_tokens_for_lm = image_tokens

        # standard VCLM with a single video
        text, labels = text[:, :-1], text[:, 1:]
        output_decoder = self.text_decoder(text.contiguous(), encoder_hidden_states=image_tokens_for_lm)
        text_tokens_logits = output_decoder.logits
        text_tokens_logits = rearrange(text_tokens_logits, 'b n c -> b c n')
        return_dict =  {'text_tokens_logits': text_tokens_logits,
                'labels': labels, 'log_dict': {}}

        return return_dict
    
    def generate_feats(self, image_tokens, tokenizer, num_return_sequences=1, prompt=None):
        with torch.no_grad():
            image_tokens = image_tokens.repeat_interleave(num_return_sequences, dim=0)
            device = image_tokens.device

            if self.args.prompt is None:
                generated_text_ids = torch.LongTensor([[tokenizer.bos_token_id]] * image_tokens.shape[0]).to(device)
            else:
                generated_text_ids = tokenizer(self.args.prompt.replace('"', '').strip()).to(device)
                cut_idx = (generated_text_ids == tokenizer.eos_token_id).nonzero(as_tuple=True)[0][-1]
                generated_text_ids = generated_text_ids[0:cut_idx]
                generated_text_ids = generated_text_ids.unsqueeze(0).repeat(image_tokens.shape[0], 1)
            condition_text_ids = generated_text_ids.clone()            

            output_decoder = self.text_decoder(condition_text_ids, encoder_hidden_states=image_tokens, output_hidden_states=True)
            hidden = output_decoder.hidden_states

            if len(hidden.shape) == 2:
                hidden = hidden.unsqueeze(0)
                condition_text_ids = condition_text_ids.unsqueeze(0)

            return hidden[:, -1]


            if prompt is None:
                condition_text_ids = torch.LongTensor([[tokenizer.bos_token_id]] * image_tokens.shape[0]).to(device)
            else:
                condition_text_ids = tokenizer(prompt).to(device)

            output_decoder = self.text_decoder(condition_text_ids, encoder_hidden_states=image_tokens, output_hidden_states=True)
            hidden = output_decoder.hidden_states

            if prompt is not None:
                eos_idxs = (condition_text_ids == tokenizer.eos_token_id).nonzero(as_tuple=False)
                eos_idxs = []
                if len(hidden.shape) == 2:
                    hidden = hidden.unsqueeze(0)
                    condition_text_ids = condition_text_ids.unsqueeze(0)
                for i in range(hidden.shape[0]):
                    try:
                        eos_idxs.append((condition_text_ids[i] == tokenizer.eos_token_id).nonzero(as_tuple=False)[1])
                    except:
                        print(prompt, condition_text_ids.shape, hidden.shape)
                        print(f"prompt does not contain enough eos tokens. too long? {prompt[i]}")
                        eos_idxs.append(torch.tensor([condition_text_ids.shape[-1] - 1]))
                eos_idxs = torch.stack(eos_idxs).squeeze()
                b_idxs = torch.arange(hidden.shape[0])
                hidden = hidden[b_idxs, eos_idxs]
            hidden = hidden.squeeze(dim=1)
            return hidden


    def generate(self, image_tokens, tokenizer, target=None, max_text_length=77, top_k=None, top_p=None,
                 num_return_sequences=1, temperature=1.0, teacher_forcing=False, early_stopping=False, arg_prompt=None):
        
        image_tokens = image_tokens.repeat_interleave(num_return_sequences, dim=0)
        device = image_tokens.device



        if (self.args.prompt is None) and (arg_prompt is None):
            generated_text_ids = torch.LongTensor([[tokenizer.bos_token_id]] * image_tokens.shape[0]).to(device)
        
        else:
            if arg_prompt is not None:
                prompt = arg_prompt
            else:
                prompt = self.args.prompt

            # prompts = [self.args.prompt.replace('"', '').strip() for _ in range(image_tokens.shape[0])]
            generated_text_ids = tokenizer(prompt.replace('"', '').strip()).to(device)
            cut_idx = (generated_text_ids == tokenizer.eos_token_id).nonzero(as_tuple=True)[0][-1]
            generated_text_ids = generated_text_ids[0:cut_idx]
            generated_text_ids = generated_text_ids.unsqueeze(0).repeat(image_tokens.shape[0], 1)

        condition_text_ids = generated_text_ids.clone()
        logits_warper = self._get_logits_warper(top_k=top_k, top_p=top_p, typical_p=None, temperature=temperature, num_beams=1)

        nlls, num_tokens = torch.zeros(image_tokens.shape[0]).to(device), torch.zeros(image_tokens.shape[0]).to(device)
        is_reach_eos = torch.zeros(image_tokens.shape[0]).bool().to(device)
        with torch.no_grad():
            for i in range(max_text_length - 1):
                output_decoder = self.text_decoder(condition_text_ids, encoder_hidden_states=image_tokens)
                decoded_token_logits = output_decoder.logits
                next_token_logits = decoded_token_logits[:, -1, :]
                if target is not None:
                    nll = F.cross_entropy(next_token_logits, target[:, i+1], ignore_index=tokenizer.pad_token_id, reduction='none')
                    nlls += nll
                    num_tokens += target[:, i+1].ne(tokenizer.pad_token_id)
                else:
                    nll = torch.special.entr(F.softmax(next_token_logits, dim=1)).sum(dim=1)
                    nlls += nll * (~is_reach_eos)
                    num_tokens += (~is_reach_eos)
                # filtered_p = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p, device=device)
                next_token_logits = logits_warper(generated_text_ids, next_token_logits)
                filtered_p = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(filtered_p, num_samples=1)
                is_reach_eos = is_reach_eos | (next_token[:, 0] == tokenizer.eos_token_id)
                if early_stopping and torch.all(is_reach_eos):
                    break

                if teacher_forcing:
                    condition_text_ids = target[:, :i+2]
                else:
                    condition_text_ids = torch.cat((generated_text_ids, next_token), dim=1)

                generated_text_ids = torch.cat((generated_text_ids, next_token), dim=1)
        if target is not None:
            return generated_text_ids, torch.exp(nlls / num_tokens)
        else:
            return generated_text_ids, torch.exp(nlls / num_tokens)

    def beam_sample(self, image_tokens, tokenizer, target=None, max_text_length=77, top_k=None, top_p=None,
                    temperature=1.0, length_penalty=1.,
                    num_beams=3, num_return_sequences=1, teacher_forcing=False, early_stopping=False):
        batch_size = image_tokens.shape[0]
        device = image_tokens.device
        input_ids = torch.ones((batch_size, 1), device=device, dtype=torch.long)
        input_ids = input_ids * tokenizer.bos_token_id

        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, num_beams * num_return_sequences).view(-1).to(device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)

        batch_beam_size, cur_len = input_ids.shape

        logits_warper = self._get_logits_warper(top_k=top_k, top_p=top_p, typical_p=None, temperature=temperature, num_beams=num_beams)

        beam_scorer = BeamSearchScorer(
            batch_size=batch_size * num_return_sequences, num_beams=num_beams,
            device=device,
            length_penalty=length_penalty,
        )
        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        beam_scores = torch.zeros((batch_size, num_beams)).to(device)
        beam_scores = beam_scores.view((batch_size * num_beams,))

        is_reach_eos = torch.zeros(batch_beam_size).bool().to(device)
        with torch.no_grad():
            for i in range(max_text_length - 1):
                output_decoder = self.text_decoder(
                    input_ids,
                    encoder_hidden_states=image_tokens.repeat_interleave(num_beams * num_return_sequences, dim=0)
                )
                decoded_token_logits = output_decoder.logits
                next_token_logits = decoded_token_logits[:, -1, :]

                next_token_scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)
                # supposed to be the line below, but ignore temporarily
                # next_token_scores_processed = logits_processor(input_ids, next_token_scores)
                next_token_scores_processed = next_token_scores
                next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(next_token_scores)
                # supposed to be the line below, but do a simple top_k+top_p temporarily
                next_token_scores = logits_warper(input_ids, next_token_scores)
                # next_token_scores = top_k_top_p_filtering(next_token_scores, top_k=top_k, top_p=top_p, device=device)

                vocab_size = next_token_scores.shape[-1]
                next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

                probs = F.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=2 * num_beams)
                next_token_scores = torch.gather(next_token_scores, -1, next_tokens)

                next_token_scores, _indices = torch.sort(next_token_scores, descending=True, dim=1)
                next_tokens = torch.gather(next_tokens, -1, _indices)

                next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
                next_tokens = next_tokens % vocab_size

                # stateless
                beam_outputs = beam_scorer.process(
                    input_ids,
                    next_token_scores,
                    next_tokens,
                    next_indices,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

                beam_scores = beam_outputs["next_beam_scores"]
                beam_next_tokens = beam_outputs["next_beam_tokens"]
                beam_idx = beam_outputs["next_beam_indices"]

                input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

                is_reach_eos = is_reach_eos | (input_ids[:, -1] == tokenizer.eos_token_id)
                if beam_scorer.is_done or torch.all(is_reach_eos):
                    break

            sequence_outputs = beam_scorer.finalize(
                input_ids,
                beam_scores,
                next_tokens,
                next_indices,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_length=max_text_length,
            )

            sequences = sequence_outputs["sequences"]
            sequence_scores = sequence_outputs["sequence_scores"]
        return sequences, sequence_scores

    def group_beam_search(self, image_tokens, tokenizer, target=None, max_text_length=77, top_k=None, top_p=None,
                          temperature=1.0, length_penalty=1.,
                          num_beams=6, num_beam_groups=3,
                          num_return_sequences=1, teacher_forcing=False, early_stopping=False):
        batch_size = image_tokens.shape[0]
        device = image_tokens.device
        input_ids = torch.ones((batch_size, 1), device=device, dtype=torch.long)
        input_ids = input_ids * tokenizer.bos_token_id

        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, num_beams).view(-1).to(device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)

        batch_beam_size, cur_len = input_ids.shape

        logits_warper = self._get_logits_warper(top_k=top_k, top_p=top_p, typical_p=None, temperature=temperature, num_beams=num_beams)

        beam_scorer = BeamSearchScorer(
            batch_size=batch_size, num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            num_beam_hyps_to_keep=num_return_sequences, device=device,
            length_penalty=length_penalty,
        )
        num_sub_beams = num_beams // num_beam_groups
        beam_scores = torch.full((batch_size, num_beams), -1e9, dtype=torch.float, device=device)
        beam_scores[:, ::num_sub_beams] = 0
        beam_scores = beam_scores.view((batch_size * num_beams,))

        is_reach_eos = torch.zeros(batch_beam_size).bool().to(device)
        with torch.no_grad():

            # predicted tokens in cur_len step
            current_tokens = torch.zeros(batch_size * num_beams, dtype=input_ids.dtype, device=device)

            # indices which will form the beams in the next time step
            reordering_indices = torch.zeros(batch_size * num_beams, dtype=torch.long, device=device)

            for i in range(max_text_length - 1):
                output_decoder = self.text_decoder(
                    input_ids,
                    encoder_hidden_states=image_tokens.repeat_interleave(num_beams, dim=0)
                )
                decoded_token_logits = output_decoder.logits

                for beam_group_idx in range(num_beam_groups):
                    group_start_idx = beam_group_idx * num_sub_beams
                    group_end_idx = min(group_start_idx + num_sub_beams, num_beams)
                    group_size = group_end_idx - group_start_idx

                    # indices of beams of current group among all sentences in batch
                    batch_group_indices = []

                    for batch_idx in range(batch_size):
                        batch_group_indices.extend(
                            [batch_idx * num_beams + idx for idx in range(group_start_idx, group_end_idx)]
                        )
                    group_input_ids = input_ids[batch_group_indices]

                    # select outputs of beams of current group only
                    next_token_logits = decoded_token_logits[batch_group_indices, -1, :]

                    next_token_scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)
                    vocab_size = next_token_scores.shape[-1]

                    # supposed to be the line below, but ignore temporarily
                    # next_token_scores_processed = logits_processor(input_ids, next_token_scores)
                    next_token_scores_processed = next_token_scores
                    next_token_scores = next_token_scores_processed + beam_scores[batch_group_indices].unsqueeze(-1)
                    next_token_scores = next_token_scores.expand_as(next_token_scores_processed)
                    next_token_scores = logits_warper(input_ids, next_token_scores)
                    # next_token_scores = top_k_top_p_filtering(next_token_scores, top_k=top_k, top_p=top_p, device=device)

                    # reshape for beam search
                    next_token_scores = next_token_scores.view(batch_size, group_size * vocab_size)

                    next_token_scores, next_tokens = torch.topk(
                        next_token_scores, 2 * group_size, dim=1, largest=True, sorted=True
                    )

                    next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
                    next_tokens = next_tokens % vocab_size

                    # stateless
                    beam_outputs = beam_scorer.process(
                        group_input_ids,
                        next_token_scores,
                        next_tokens,
                        next_indices,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        beam_indices=None
                    )
                    beam_scores[batch_group_indices] = beam_outputs["next_beam_scores"]
                    beam_next_tokens = beam_outputs["next_beam_tokens"]
                    beam_idx = beam_outputs["next_beam_indices"]

                    input_ids[batch_group_indices] = group_input_ids[beam_idx]
                    group_input_ids = torch.cat([group_input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
                    current_tokens[batch_group_indices] = group_input_ids[:, -1]
                    reordering_indices[batch_group_indices] = (
                        num_beams * torch.div(beam_idx, group_size, rounding_mode="floor") + group_start_idx + (beam_idx % group_size)
                    )

                input_ids = torch.cat([input_ids, current_tokens.unsqueeze(-1)], dim=-1)

                is_reach_eos = is_reach_eos | (input_ids[:, -1] == tokenizer.eos_token_id)
                if beam_scorer.is_done or torch.all(is_reach_eos):
                    break

            sequence_outputs = beam_scorer.finalize(
                input_ids,
                beam_scores,
                next_tokens,
                next_indices,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_length=max_text_length,
                beam_indices=None,
            )

            sequences = sequence_outputs["sequences"]
            sequence_scores = sequence_outputs["sequence_scores"]
        return sequences, sequence_scores

    def _get_logits_warper(
        self, top_k=None, top_p=None, typical_p=None,
        temperature=None, num_beams=None, renormalize_logits=None,
    ):
        top_k = top_k if top_k is not None else 0
        top_p = top_p if top_p is not None else 1.0
        typical_p = typical_p if typical_p is not None else 1.
        temperature = temperature if temperature is not None else 1.
        warpers = LogitsProcessorList()

        if temperature is not None and temperature != 1.0:
            warpers.append(TemperatureLogitsWarper(temperature))
        if top_k is not None and top_k != 0:
            warpers.append(TopKLogitsWarper(top_k=top_k, min_tokens_to_keep=(2 if num_beams > 1 else 1)))
        if top_p is not None and top_p < 1.0:
            warpers.append(TopPLogitsWarper(top_p=top_p, min_tokens_to_keep=(2 if num_beams > 1 else 1)))
        if typical_p is not None and typical_p < 1.0:
            warpers.append(TypicalLogitsWarper(mass=typical_p, min_tokens_to_keep=(2 if num_beams > 1 else 1)))
        # `LogitNormalization` should always be the last logit processor, when present
        if renormalize_logits is True:
            warpers.append(LogitNormalization())
        return warpers
