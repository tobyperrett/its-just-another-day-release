import argparse
import os.path as osp
from collections import OrderedDict
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms._transforms_video as transforms_video

from lavila.data import datasets
from lavila.data.video_transforms import Permute
from lavila.models import models, prompt_predictor
from lavila.utils.preprocess import generate_tokenizer
from lavila.utils import distributed as dist_utils
import itertools

import torch.nn.functional as F

from einops import rearrange

from main_infer_narrator import do_caption

import random

import pickle
import wandb

import os
os.environ['WANDB_MODE'] = 'offline'
# import networkx as nx
import inspect




def get_args_parser():
    parser = argparse.ArgumentParser(description='lavila infer narrator', add_help=False)
    parser.add_argument('--dataset', default='ego4d', type=str, choices=['ego4d', 'tlm'])
    parser.add_argument('--root', default='storage/video_384_30fps_300s', type=str, help='path to dataset root')
    parser.add_argument('--chunk_len', default=300, type=int, help='chunk length of mp4s')
    parser.add_argument('--metadata', default='storage/train_preds.pkl', type=str, help='path to metadata file')
    parser.add_argument('--metadata_sets', default='storage/amin_clip_P1_OnlinePerVid_p10_e0.pkl', type=str, help='path to metadata file')
    parser.add_argument('--output-dir', default='output', type=str, help='output dir')
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--use-half', action='store_true')
    parser.add_argument('--clip-length', default=4, type=int, help='clip length')
    parser.add_argument('--clip-stride', default=16, type=int, help='clip stride')
    parser.add_argument('--emb_resume', default='storage/clip_openai_timesformer_large.narrator_rephraser.ep_0003.md5sum_c89337.pth', type=str, help='path to latest checkpoint')
    parser.add_argument('--eval_emb_resume', default='storage/clip_openai_timesformer_large_336px_distilbert_base.baseline.ep_0003.pth', type=str, help='path to latest checkpoint')
    parser.add_argument('--cap_resume', default='storage/vclm_openai_timesformer_large_336px_gpt2_xl.pt_ego4d.jobid_246897.ep_0003.md5sum_443263.pth', type=str, help='path to latest checkpoint')
    parser.add_argument('--pp_resume', default='storage/arch10.pkl', type=str, help='path to latest checkpoint')
    parser.add_argument('--caption-sample', default='multinomial_sample',
                        choices=['multinomial_sample', 'beam_sample', 'group_beam_search'])
    parser.add_argument('--caption-top-k', default=None, type=int)
    parser.add_argument('--caption-top-p', default=0.95, type=float)
    parser.add_argument('--caption-num-beams', default=1, type=int)
    parser.add_argument('--caption-num-beam-groups', default=1, type=int)
    parser.add_argument('--caption-temperature', default=0.7, type=float)
    parser.add_argument('--caption-length-penalty', default=1.0, type=float)
    parser.add_argument('--caption-num-return-sequences', default=1, type=int)
    parser.add_argument('--caption-max-len', default=77, type=int)
    parser.add_argument('--caption-early-stop', action='store_true', help='early stopping to save computation')
    # System
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                        help='number of data loading workers per process')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str)
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')

    parser.add_argument('--return_original_narration', default=1, type=int)
    parser.add_argument('--return_positive', default=0, type=int)
    parser.add_argument('--return_negative', default=0, type=int)
    parser.add_argument('--before_after_choice', default="random",choices=['random', 'before', 'after'], type=str, help="return before, after, or random")
    parser.add_argument('--dataloader_text', default="narration", choices=["caption_lav-base", "narration"], help='which type of text to load from dataset')

    # options for loading multiple similar videos
    parser.add_argument('--load_similar_videos', default=1, type=int)
    parser.add_argument('--load_similar_videos_range', default=[-1], help="one int for constant, two ints for range. -1 to load all in the set, which can vary for e.g. long videos")

    #override variable clip length
    parser.add_argument('--fixed_clip_len', default=5, type=float, help="fixed clip_length")
    parser.add_argument('--extend', default="both", type=str, help="how to extend fixed clip length")
    parser.add_argument('--prepend', default=None, choices=[None, "cur_verb"], help="Prepend a verb to the narration")
    parser.add_argument('--prompt', default=None)

    parser.add_argument('--caption_num_return_sequences', default = 1, type=int)
    parser.add_argument('--caption_sample', default = 'multinomial_sample', type=str)
    parser.add_argument('--n_sets_start', default=0, type=int, help="stop after a certain number of sets")
    parser.add_argument('--n_sets', default=300, type=int, help="stop after a certain number of sets")
    parser.add_argument('--n_img_q', default=256, type=int, help="num_img_queries in narrator")
    parser.add_argument('--estimator', default=None, type=str, help="estimator")
    parser.add_argument('--uct_type', default=None, type=str, help="estimator")
    parser.add_argument('--sim_type', default=["rand"], type=str, help="estimator")
    parser.add_argument('--debug', default=0, type=int, help="threshold for unique caption")
    parser.add_argument('--pp_threshold', default=2.0, type=float)
    parser.add_argument('--min_group_sim', default=0.7, type=float)
    parser.add_argument('--min_vt_sim', default=0.0, type=float)
    parser.add_argument('--temporal_offset', default=5, type=int)
    parser.add_argument('--max_offset', default=0, type=int)
    parser.add_argument('--ignore_prompt', default=0, type=int, help="ignore prompt to just use standard lavila captioning")
    parser.add_argument('--no_cap_default', default="best_margin", type=str, help="if video is not predicted as unique, default to the initial prompt")

    parser.add_argument('--pp_vt_prob', default="mean", type=str, help="canculate difference between max and second max from v and t")
    parser.add_argument('--comb_cs', default="mean", type=str, help="how to combine in method")
    parser.add_argument('--comb_eval', default="mean", type=str, help="combine at evaluation")
    parser.add_argument('--comb_maxp', default=3, type=int, help="max cardinality of prompt combinations")
    parser.add_argument('--enforce_amax', default=1, type=int, help="assume the margin is always positive")
    parser.add_argument('--priority', default="none", type=str, help="order of preference for going through prompt combinations")
    parser.add_argument('--n_rand_prompts', default=None, type=int, help="use for dropping randomly dropping prompts at inference")

    parser.add_argument('--gt', default=0, type=int, help="load gt prompt preds")

    parser.add_argument('--emb_max_batch', default=32)
    parser.add_argument('--pp_max_batch', default=8192)
    parser.add_argument('--work_device', default="cuda", type=str)

    # options for saving output
    parser.add_argument('--wandb', default=0, type=int)
    parser.add_argument('--frames', default=4, type=int)
    parser.add_argument('--fps', default=1, type=int)
    parser.add_argument('--video_size', default=224, type=int)
    parser.add_argument('--wandb_project', default="vis3", type=str)
    parser.add_argument('--wbr', default=None, type=str)

    parser.add_argument('--d_model', default=256, type=int, help="num prompts")
    parser.add_argument('--d_ff', default=1024, type=int, help="num prompts")
    parser.add_argument('--pp_feat_str', default="vis_feats", type=str, help="num prompts")
    parser.add_argument('--pp_nh', default=4, type=int, help="num prompts")
    parser.add_argument('--pp_nl', default=2, type=int, help="num prompts")
    
    return parser


# dataloader for timeloop movies, which have pre-computed features
class TLMFeatures():
    def __init__(self, args) -> None:
        self.args = args

        self.n_prompts = 10
        self.blank_videos = torch.zeros(self.n_prompts, 4, 3, 28, 28)

        self.feats = torch.load(osp.join(self.args.root))
        
        self.set_keys = list(self.feats.keys())

    def extract_caption_single(self, vid_key, prompt_id=None, offset=0, set_key=0):
        if prompt_id is None:
            prompt_id = -1
        tf = self.feats[set_key][vid_key]['textual'][int(offset / self.args.temporal_offset)]['answer'][prompt_id]
        return tf

    def get_vis_embeddings_pp(self, offset=0, set_key=0):
        vf = [self.feats[set_key][k]['visual'][int(offset / self.args.temporal_offset)]['ViT-L-14'] for k in self.feats[set_key].keys()]
        vf = torch.stack(vf, dim=0)
        vf = torch.mean(vf, dim=-2)
        return vf, self.blank_videos

    def get_vis_embeddings_eval(self, max_offset=0, set_key=0):
        set_size = len(self.feats[set_key][list(self.feats[set_key].keys())[0]]['visual'])
        all_vf = []
        for i in range(set_size):
            vf = [self.feats[set_key][k]['visual'][i]['internvideo'] for k in self.feats[set_key].keys()]
            vf = torch.stack(vf, dim=0)
            all_vf.append(vf)

        vf = torch.stack(all_vf, dim=0)
        vf = torch.mean(vf, dim=-2)
        return vf

    def get_text_embedding_eval(self, vid_key=0, offset=0, set_key=0, prompt_id=None):
        if prompt_id is None:
            prompt_id = -1
        tf = self.feats[set_key][vid_key]['textual'][int(offset / self.args.temporal_offset)]['text_feature_internvideo'][prompt_id]
        return tf


class UniqueCaptioner():
    def __init__(self, args) -> None:
        self.args = args

        if self.args.dataset == "ego4d":
        
            default_size = 384 if "384" in self.args.root else 224
            self.default_transform = transforms.Compose([
                Permute([3, 0, 1, 2]),  # T H W C -> C T H W
                transforms.Resize(default_size),
                transforms.CenterCrop(default_size)
            ])

            self.emb_model, self.emb_tokenizer, self.emb_val_transform, self.emb_val_transform_norm = self.load_embedding_model(self.args.emb_resume)
            self.eval_emb_model, self.eval_emb_tokenizer, self.eval_emb_val_transform, self.eval_emb_val_transform_norm = self.load_embedding_model(self.args.eval_emb_resume)
            self.cap_model, self.cap_tokenizer, self.cap_val_transform, self.cap_val_transform_norm = self.load_embedding_model(self.args.cap_resume)
            self.load_embedding_dataset()
            self.cap_val_dataset = self.emb_val_dataset

            self.load_prompt_predictor()
            self.prompts = [
                "#C C picks ",
                "#C C holds ",
                "#C C looks at ",
                "#C C moves the ",
                "#C C walks towards the ",
                "#C C walks around the ",
                "#C C goes past the ",
                "#C C is in the ",
                "#O the man ",
                "#O the woman ",
            ]
        elif self.args.dataset == "tlm":
            self.tlmfeats = TLMFeatures(self.args)
            self.prompts = [
                "#C C picks ",
                "#C C holds ",
                "#C C looks at ",
                "#C C moves the ",
                "#C C walks towards the ",
                "#C C walks around the ",
                "#C C goes past the ",
                "#C C is in the ",
                "#O the man ",
                "#O the woman ",
            ]
            self.load_prompt_predictor()


    def load_embedding_dataset(self):
        self.emb_val_dataset = datasets.VideoCaptionDatasetCLIP(
            args.dataset,
            args.root,
            args.metadata,
            # transform=self.emb_val_transform,
            transform=self.default_transform,
            is_training=False,
            tokenizer=self.emb_tokenizer,
            clip_length=args.clip_length,
            clip_stride=args.clip_stride,
            sparse_sample=False,
            subsample_stride=None,
            return_original_narration=args.return_original_narration,
            return_positive=args.return_positive,
            return_negative=args.return_negative,
            before_after_choice=args.before_after_choice,
            args=args
        )
    def load_captioning_dataset(self):
        self.cap_val_dataset = datasets.VideoCaptionDatasetCLIP(
            args.dataset,
            args.root,
            args.metadata,
            # transform=self.cap_val_transform,
            transform=self.default_transform,
            is_training=False,
            tokenizer=self.cap_tokenizer,
            clip_length=args.clip_length,
            clip_stride=args.clip_stride,
            sparse_sample=False,
            subsample_stride=None,
            return_original_narration=args.return_original_narration,
            return_positive=args.return_positive,
            return_negative=args.return_negative,
            before_after_choice=args.before_after_choice,
            args=args
        )
    def load_embedding_model(self, resume):
        args = self.args

        if resume:
            ckpt_path = resume
        elif osp.isfile(osp.join(args.output_dir, 'checkpoint_best.pt')):
            ckpt_path = osp.join(args.output_dir, 'checkpoint_best.pt')
        else:
            raise Exception('no checkpoint found')

        ckpt = torch.load(ckpt_path, map_location='cpu')
        state_dict = OrderedDict()
        for k, v in ckpt['state_dict'].items():
            state_dict[k.replace('module.', '')] = v

        old_args = ckpt['args']
        emb_tokenizer = generate_tokenizer(old_args.model)
        model = getattr(models, old_args.model)(
            text_use_cls_token=old_args.use_cls_token,
            gated_xattn=old_args.gated_xattn if hasattr(old_args, 'gated_xattn') else None,
            timesformer_gated_xattn=old_args.timesformer_gated_xattn if hasattr(old_args, 'timesformer_gated_xattn') else None,
            num_frames=old_args.clip_length,
            drop_path_rate=0,
            tokenizer=emb_tokenizer,
            args=args
        )
        model.cuda()
        result = model.load_state_dict(state_dict, strict=False)

        torch.backends.cudnn.benchmark = True

        crop_size = 224 if '336PX' not in old_args.model else 336
        val_transform = transforms.Compose([
            # Permute([3, 0, 1, 2]),  # T H W C -> C T H W
            transforms.Resize(crop_size),
            transforms.CenterCrop(crop_size),
        ])
        val_transform_norm = transforms.Compose([
             (transforms_video.NormalizeVideo(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]) if 'OPENAI' not in old_args.model else
                transforms_video.NormalizeVideo(mean=[108.3272985, 116.7460125, 104.09373615000001], std=[68.5005327, 66.6321579, 70.32316305])),
        ])           

        model.eval()
        if args.use_half:
            model.half()

        return model, emb_tokenizer, val_transform, val_transform_norm

    def load_captioner(self):
        args = self.args

        if args.cap_resume:
            ckpt_path = args.cap_resume
        elif osp.isfile(osp.join(args.output_dir, 'checkpoint_best.pt')):
            ckpt_path = osp.join(args.output_dir, 'checkpoint_best.pt')
        else:
            raise Exception('no checkpoint found')

        ckpt = torch.load(ckpt_path, map_location='cpu')
        state_dict = OrderedDict()
        for k, v in ckpt['state_dict'].items():
            state_dict[k.replace('module.', '')] = v

        old_args = ckpt['args']
        self.cap_tokenizer = generate_tokenizer(old_args.model)
        model = getattr(models, old_args.model)(
            text_use_cls_token=old_args.use_cls_token,
            gated_xattn=old_args.gated_xattn if hasattr(old_args, 'gated_xattn') else None,
            timesformer_gated_xattn=old_args.timesformer_gated_xattn if hasattr(old_args, 'timesformer_gated_xattn') else None,
            num_frames=old_args.clip_length,
            drop_path_rate=0,
            tokenizer=self.cap_tokenizer,
            args=args
        )
        model.cuda()
        model.load_state_dict(state_dict, strict=False)

        crop_size = 224 if '336PX' not in old_args.model else 336
        self.cap_val_transform = transforms.Compose([
            # Permute([3, 0, 1, 2]),  # T H W C -> C T H W
            transforms.Resize(crop_size),
            transforms.CenterCrop(crop_size),
            (transforms_video.NormalizeVideo(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]) if 'OPENAI' not in old_args.model else
                transforms_video.NormalizeVideo(mean=[108.3272985, 116.7460125, 104.09373615000001], std=[68.5005327, 66.6321579, 70.32316305])),
        ])

        model.eval()
        if args.use_half:
            model.half()
        self.cap_model = model
    def load_prompt_predictor(self):
        args = self.args

        if args.pp_resume:
            ckpt_path = args.pp_resume
        elif osp.isfile(osp.join(args.output_dir, 'checkpoint_best.pt')):
            ckpt_path = osp.join(args.output_dir, 'checkpoint_best.pt')
        else:
            raise Exception('no checkpoint found')

        ckpt = torch.load(ckpt_path, map_location='cpu')
        state_dict = OrderedDict()
        for k, v in ckpt['state_dict'].items():
            state_dict[k.replace('module.', '')] = v

        old_args = ckpt['args']
        model = getattr(prompt_predictor, old_args.model)(
            args=old_args
        )
        model.cuda()
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        self.pp_model = model
    def get_all_set_keys(self):
        if self.args.dataset == "ego4d":
            return self.emb_val_dataset.set_keys
        elif self.args.dataset == "tlm":
            return self.tlmfeats.set_keys
        
    def get_vis_embeddings(self, set_k, temporal_offset=0, cpu_frames=None, val_transform=None, norm_transform=None, emb_model=None, feat_str="vis_feats"):
        if self.args.dataset == "ego4d":
            norm_frames = []
            for i in range(cpu_frames.shape[0]):
                trans_frames = None
                if val_transform is not None:
                    trans_frames = val_transform(cpu_frames[i])
                if norm_transform is not None:
                    trans_frames = norm_transform(trans_frames)
                if trans_frames is not None:
                    norm_frames.append( trans_frames )

            if norm_frames:
                frames = torch.stack(norm_frames)

            bb = frames.shape[0]
            if bb > self.args.emb_max_batch:
                frames = torch.split(frames, self.args.emb_max_batch, dim=0)
                embs = []
                for idx, f in enumerate(frames):
                    f = f.cuda()
                    if feat_str == "enc_vis_feats":
                        f_emb = emb_model.encode_image(f, return_features=True)[1]
                    else:
                        f_emb = emb_model.encode_image(f)
                    embs.append(f_emb)
                embs = torch.cat(embs, dim=0)
            else:
                frames = frames.cuda()
                if feat_str == "enc_vis_feats":
                    embs = emb_model.encode_image(frames, return_features=True)[1]
                else:
                    embs = emb_model.encode_image(frames)

        elif self.args.dataset == "tlm":
            embs, cpu_frames = self.tlmfeats.get_vis_embeddings_pp(temporal_offset, set_k)
            embs = embs.cuda()
        return embs
    
    def get_frames_only(self, set_k, temporal_offset=0):
        if self.args.dataset == "ego4d":
            cpu_frames = self.emb_val_dataset.__getitem__(set_k, i_is_key=True, offset_s=temporal_offset)["frames"]

        elif self.args.dataset == "tlm":
            raise NotImplementedError("tlm does not support frames only")
        return 0, cpu_frames
    
    def get_text_embeddings(self, texts, vid_key=0, temporal_offset=0, set_key=0, prompt_id=None):

        if self.args.dataset == "ego4d":
            text_tokens = self.eval_emb_tokenizer(texts)
            if isinstance(text_tokens, torch.Tensor):
                text_tokens = text_tokens.cuda()
                if len(text_tokens.shape) == 1:
                    text_tokens = text_tokens.unsqueeze(0)
                text_embs = self.eval_emb_model.encode_text(text_tokens)
            else:
                text_tokens, mask = text_tokens
                text_tokens = text_tokens.cuda()
                mask = mask.cuda()
                if len(text_tokens.shape) == 1:
                    text_tokens = text_tokens.unsqueeze(0)
                    mask = mask.unsqueeze(0)
                text_embs = self.eval_emb_model.encode_text(text_tokens, attention_mask=mask)               

        elif self.args.dataset == "tlm":
            text_embs = self.tlmfeats.get_text_embedding_eval(vid_key, temporal_offset, set_key, prompt_id=prompt_id).unsqueeze(0).cuda()

        return text_embs
    

    def extract_caption_single(self, vid_keys, prompt_id=None, temporal_offset=0, set_key=0, frames=None):

        if self.args.dataset == "ego4d":
            if frames is None:
                frames = [self.cap_val_dataset.__get_single_item__(k, index_is_key=True, offset_s=temporal_offset)[0] for k in vid_keys]
                frames = torch.stack(frames).cuda()
            else:
                frames = self.cap_val_transform(frames)
                frames = self.cap_val_transform_norm(frames).cuda().unsqueeze(0)

            image_tokens, _ = self.cap_model.encode_image(frames)

            if prompt_id is not None:
                prompt = self.prompts[prompt_id]
            else:
                prompt = None

            captions = do_caption(args, self.cap_model, image_tokens, self.cap_tokenizer, sample_keys=None, val_loader=None, target=None, return_text_only=True, prompt=prompt)
        elif self.args.dataset == "tlm":

            captions = self.tlmfeats.extract_caption_single(vid_keys[0], prompt_id, temporal_offset, set_key=set_key)
            captions = [captions]

        return captions
    
    def extract_captions_batch(self):
        raise NotImplementedError("batch captioning not implemented")



    def group_get_metrics(self, is_unique_preds, all_vid_keys, chosen_offsets, chosen_cap_emb, all_vis_emb):
        cs_cols = []
        for k in all_vid_keys:
            cs_per_vid = []
            for j in range(len(chosen_offsets[k])):
                v_e = all_vis_emb[chosen_offsets[k][j]]
                t_e = chosen_cap_emb[k][j]
                cs_col = F.cosine_similarity(v_e.unsqueeze(-3), t_e.unsqueeze(-2).unsqueeze(-2), dim=-1).transpose(-1, -2)
                cs_per_vid.append(cs_col)

            cs_per_vid = torch.stack(cs_per_vid, dim=-1)

            if self.args.comb_eval == "mean":
                cs_per_vid = torch.mean(cs_per_vid, dim=-1, keepdim=False)
            elif self.args.comb_eval == "min":
                cs_per_vid = torch.min(cs_per_vid, dim=-1, keepdim=False)[0]
            elif self.args.comb_eval == "threshold":
                cs_per_vid = torch.mean(torch.where(cs_per_vid > self.args.min_group_sim, torch.tensor(1.0, device=cs_per_vid.device), torch.tensor(0.0, device=cs_per_vid.device)), dim=-1)

            cs_cols.append(cs_per_vid)

        # merge for full cs matrix
        cs = torch.cat(cs_cols, dim=-1)

        # accuracy
        n_u_preds = float(sum([v for v in is_unique_preds.values()]))
        diag_labs = torch.arange(cs.shape[0])
        v_preds = torch.argmax(cs, dim=-1).cpu()
        t_preds = torch.argmax(cs, dim=-2).cpu()

        v_acc = torch.mean((v_preds == diag_labs).float()) * 100
        t_acc = torch.mean((t_preds == diag_labs).float()) * 100

        v_acc_w1 = torch.mean((torch.abs(v_preds - diag_labs) <= 1.0).float()) * 100
        v_acc_w2 = torch.mean((torch.abs(v_preds - diag_labs) <= 2.0).float()) * 100
        t_acc_w1 = torch.mean((torch.abs(t_preds - diag_labs) <= 1.0).float()) * 100
        t_acc_w2 = torch.mean((torch.abs(t_preds - diag_labs) <= 2.0).float()) * 100

        vt_acc = 0.0
        vt_pred_only = 0.0
        for i in range(cs.shape[0]):
            if v_preds[i] == i and t_preds[i] == i:
                vt_acc += 1
                if is_unique_preds[all_vid_keys[i]]:
                    vt_pred_only += 1
        vt_acc = vt_acc / cs.shape[0] * 100

        seconds = (all_vis_emb.shape[0] - 1) * self.args.temporal_offset
        n_sentences = float(sum([len(chosen_offsets[k]) for k in all_vid_keys])) / len(all_vid_keys)
        set_size = len(all_vid_keys)

        set_size = len(all_vid_keys)

        top10_v = torch.topk(cs, set_size, dim=-1)[1].cpu()
        top10_t = torch.topk(cs, set_size, dim=-2)[1].t().cpu()

        v_acc_2 = sum([float((diag_labs[i] in top10_v[i][:2])) for i in range(diag_labs.shape[0])]) / diag_labs.shape[0]  * 100
        v_acc_3 = sum([float((diag_labs[i] in top10_v[i][:3])) for i in range(diag_labs.shape[0])]) / diag_labs.shape[0]  * 100
        t_acc_2 = sum([float((diag_labs[i] in top10_t[i][:2])) for i in range(diag_labs.shape[0])]) / diag_labs.shape[0]  * 100
        t_acc_3 = sum([float((diag_labs[i] in top10_t[i][:3])) for i in range(diag_labs.shape[0])]) / diag_labs.shape[0]  * 100
        
        avg_max_offset = sum([max(chosen_offsets[k]) for k in all_vid_keys]) / float(len(all_vid_keys))

        metric_dict = {
            "v": v_acc,
            "t": t_acc,
            "vtc1": vt_acc,
            "vtavg": (v_acc + t_acc) / 2,
            "v2": v_acc_2,
            "v3": v_acc_3,
            "t2": t_acc_2,
            "t3": t_acc_3,
            "vw1": v_acc_w1,
            "vw2": v_acc_w2,
            "tw1": t_acc_w1,
            "tw2": t_acc_w2,
            "vt_pred": vt_pred_only,
            "n_u_att": n_u_preds,
            "n_sen": n_sentences,
            "offset": avg_max_offset,
            "set_s": set_size,
            "secs": seconds,
        }
    

        print(metric_dict)
        return metric_dict, cs


    def search_combinations_bf(self, cs, combinations, unique_idxs, found_unique, best_margin):
        pp, vv, _ = cs.shape

        best_comb_idx = torch.ones(vv, dtype=torch.long) * -1
        best_comb_prob = torch.ones(vv, dtype=torch.float) * -10

        for comb_idx, comb in enumerate(combinations):
            if self.args.comb_cs == "mean":
                comb_cs = torch.mean(cs[comb, :, :], dim=0)
            elif self.args.comb_cs == "min":
                comb_cs = torch.min(cs[comb, :, :], dim=0)[0]
            else:
                raise NotImplementedError()

            # get difference between diagonal, and highest off-diagonal
            eye = torch.eye(vv, device=cs.device)
            diag_cs = comb_cs * eye
            v_top_val, v_top_idx = torch.topk(comb_cs - diag_cs, 1, dim=-2)
            t_top_val, t_top_idx = torch.topk(comb_cs - diag_cs, 1, dim=-1)
            t_top_val = t_top_val.transpose(-1, -2)
            t_top_idx = t_top_idx.transpose(-1, -2)
            diag_entries = torch.sum(diag_cs, dim=-1)
            v_diff = diag_entries - v_top_val[0]
            t_diff = diag_entries - t_top_val[0]

            if self.args.pp_vt_prob == "mean":
                diff = (v_diff + t_diff) / 2
            elif self.args.pp_vt_prob == "min":
                diff = torch.min(v_diff, t_diff)

            for i in range(vv):
                if found_unique[i] == 1:
                    continue
                if diff[i] > best_comb_prob[i]:
                    best_comb_prob[i] = diff[i]
                    best_comb_idx[i] = comb_idx              

        for i in range(vv):
            if found_unique[i] == 1:
                continue
            if best_comb_prob[i] > best_margin[i]:
                unique_idxs[i] = combinations[best_comb_idx[i]]
                best_margin[i] = best_comb_prob[i]
            if best_comb_prob[i] > self.args.pp_threshold:
                found_unique[i] = 1

        return unique_idxs, found_unique, best_margin
    
    def identify_best_combination(self, all_vid_keys, cs, pp):
        np = pp
        pp, vv, _ = cs.shape
        nt = int(pp) / int(np)
        nt = int(nt)

        unique_idxs = {i: [] for i in range(len(all_vid_keys))}
        found_unique = {i: 0 for i in range(len(all_vid_keys))}
        best_margin = {i: -100 for i in range(len(all_vid_keys))}

        prompt_idxs_to_use = [i for i in range(pp)]

        if self.args.n_rand_prompts is not None:
            prompt_idxs_to_use = random.sample(prompt_idxs_to_use, self.args.n_rand_prompts)
            prompt_idxs_to_use.sort()

        priority_combinations = []

        if self.args.priority == "tuple":
            for i in range(1, self.args.comb_maxp + 1):
                combinations = list(itertools.combinations(prompt_idxs_to_use, i))
                combinations = [sorted(c) for c in  combinations]
                priority_combinations.append(combinations)
        elif self.args.priority == "none":
            all_combinations = []
            for i in range(1, self.args.comb_maxp + 1):
                combinations = list(itertools.combinations(prompt_idxs_to_use, i))
                combinations = [sorted(c) for c in  combinations]
                all_combinations.extend(combinations)
            priority_combinations.append(all_combinations)

        else:
            raise NotImplementedError("priority not implemented yet")

        for combinations in priority_combinations:
            unique_idxs, found_unique, best_margin = self.search_combinations_bf(cs, combinations, unique_idxs, found_unique, best_margin)

        unique_idxs = {all_vid_keys[i]: unique_idxs[i] for i in unique_idxs.keys()}
        
        found_unique = {all_vid_keys[i]: found_unique[i] for i in found_unique.keys()}
        best_margin = {all_vid_keys[i]: best_margin[i] for i in best_margin.keys()}

        return found_unique, unique_idxs, best_margin


    def comb_gather_preds(self, vis_emb):

        data_dict = {self.args.pp_feat_str: vis_emb.unsqueeze(0)}

        if "pp_max_batch" in list(inspect.signature(self.pp_model.forward).parameters.keys()):
            data_dict = self.pp_model(data_dict, pp_max_batch=self.args.pp_max_batch)
        else:
            data_dict = self.pp_model(data_dict)    

        cs_bpvv = data_dict["output_cs_bpvv"]
        bb, pp, vv, _ = cs_bpvv.shape
        assert bb == 1
        cs_pvv = cs_bpvv[0]
        return cs_pvv
    
    def comb_run_offline(self, all_vid_keys, all_cs):
        # all_cs is t p v v
        tt, pp, vv, _ = all_cs.shape
        all_cs = rearrange(all_cs, 't p v1 v2 -> (t p) v1 v2')

        is_unique_preds, chosen_prompts, best_margins = self.identify_best_combination(all_vid_keys, all_cs, pp)
        chosen_offsets = {k: [] for k in all_vid_keys}

        for k, vid_p in chosen_prompts.items():
            vid_prompts = []
            vid_offsets = []
            for p in vid_p:
                vid_prompts.append(p % pp)
                vid_offsets.append(p // pp)
            chosen_prompts[k] = vid_prompts
            chosen_offsets[k] = vid_offsets

        return is_unique_preds, chosen_offsets, chosen_prompts, best_margins
    
    def comb_run_set(self, set_k):
        print(set_k)
        with torch.no_grad():
            if self.args.dataset == "ego4d":
                all_vid_keys = self.emb_val_dataset.sets[set_k]
            elif self.args.dataset == "tlm":
                all_vid_keys = list(self.tlmfeats.feats[set_k].keys())
            if self.args.gt:
                assert self.args.max_offset == 0
                gt = [self.emb_val_dataset.samples[vk]["prompt_id_gt"] for vk in all_vid_keys]
                gt = torch.tensor(gt).cuda()
            
            all_vis_emb = []
            all_eval_vis_emb = []
            all_frames = []
            all_cs = []

            # loop through all offsets and extract embeddings and frames
            current_offset = 0
            while current_offset * self.args.temporal_offset <= self.args.max_offset:
                if self.args.dataset == "ego4d":
                    _, frames = self.get_frames_only(set_k, current_offset * self.args.temporal_offset)
                    vis_emb = self.get_vis_embeddings(set_k, current_offset * self.args.temporal_offset, cpu_frames=frames, val_transform=self.emb_val_transform, norm_transform=self.emb_val_transform_norm, emb_model=self.emb_model, feat_str=self.args.pp_feat_str)
                    eval_vis_emb = self.get_vis_embeddings(set_k, current_offset * self.args.temporal_offset, cpu_frames=frames, val_transform=self.eval_emb_val_transform, norm_transform=self.eval_emb_val_transform_norm, emb_model=self.eval_emb_model, feat_str="vis_feats")
                else:
                    frames = torch.zeros(1)
                    vis_emb = self.get_vis_embeddings(set_k, current_offset * self.args.temporal_offset)
                    eval_vis_emb = torch.zeros(1)

                all_vis_emb.append(vis_emb)
                all_eval_vis_emb.append(eval_vis_emb)
                all_frames.append(frames)

                cs = self.comb_gather_preds(vis_emb)

                all_cs.append(cs)

                current_offset += 1
            current_offset -= 1

            all_vis_emb = torch.stack(all_vis_emb, dim=0)
            all_eval_vis_emb = torch.stack(all_eval_vis_emb, dim=0)

            all_frames = torch.stack(all_frames, dim=0)

            # t p v v
            all_cs = torch.stack(all_cs, dim=0)

            # choose best set of prompts
            is_unique_preds, chosen_offsets, chosen_prompts, _ = self.comb_run_offline(all_vid_keys, all_cs)

            chosen_captions = {k: [] for k in all_vid_keys}
            chosen_cap_emb = {k: [] for k in all_vid_keys}

            # set default values for videos that are not unique
            for i, k in enumerate(all_vid_keys):
                if not is_unique_preds[k]:
                    if self.args.no_cap_default == "none":
                        chosen_cap_emb[k] = [torch.zeros(self.args.d_model).cuda()]
                        chosen_offsets[k] = [0]
                    elif self.args.no_cap_default == "best_margin":
                        # prompts and offsets should be set to those with the best margin
                        pass
                    elif self.args.no_cap_default == "lav":
                        chosen_prompts[k] = [None]
                        chosen_offsets[k] = [current_offset]
                    else:
                        raise NotImplementedError(f"{self.args.no_cap_default} not implemented")

            # extract captions and cap embeddings for each predicted unique prompt
            for i, k in enumerate(all_vid_keys):
                if len(chosen_cap_emb[k]) == 0:
                    for j in range(len(chosen_prompts[k])):
                        if self.args.dataset == "ego4d":
                            frames_to_cap = all_frames[int(chosen_offsets[k][j]),i,:,:,:]
                        else:
                            frames_to_cap = None
                        caption = self.extract_caption_single([k], prompt_id=chosen_prompts[k][j], temporal_offset=chosen_offsets[k][j] * self.args.temporal_offset, set_key=set_k, frames=frames_to_cap)[0]
                        chosen_captions[k].append(caption)

                        cap_emb_single = self.get_text_embeddings(caption, vid_key=k, temporal_offset=chosen_offsets[k][j] * self.args.temporal_offset, set_key=set_k, prompt_id=chosen_prompts[k][j])[0]
                        chosen_cap_emb[k].append(cap_emb_single)

            # save video frames for wandb
            if self.args.wandb:
                row = []
                for i, k in enumerate(all_vid_keys):
                    frames = all_frames[:,i,:,:,:,:]
                    frames = rearrange(frames, 'o c t w h -> (o t) c w h')
                    frames = transforms.functional.resize(frames, args.video_size)
                    blank_frame = frames[0].clone() * 0
                    blank_frame = blank_frame.unsqueeze(0)
                    frames = torch.cat([blank_frame, frames], dim=0)
                    frames = frames.numpy()

                    wandb_captions = []
                    for o_idx, offset in enumerate(chosen_offsets[k]):
                        wandb_captions.append(f"{offset}: {chosen_captions[k][o_idx]}")

                    row.append(wandb.Video(frames, fps=args.fps, format="webm", caption=',  '.join(wandb_captions)))
                wandb.log({set_k: row})

            if self.args.dataset == "ego4d":
                pass
            elif self.args.dataset == "tlm":
                all_eval_vis_emb = self.tlmfeats.get_vis_embeddings_eval(set_key=set_k, max_offset=current_offset).cuda()

            print(all_eval_vis_emb.shape)

            metric_dict, cs_eval = self.group_get_metrics(is_unique_preds, all_vid_keys, chosen_offsets, chosen_cap_emb, all_eval_vis_emb)

            pred_dict = {"preds": {k: {"caption": chosen_captions[k], "offset": chosen_offsets[k], "prompts": chosen_prompts[k]} for k in all_vid_keys},
                         "metrics": metric_dict,
                         "cm": cs_eval.cpu().numpy(),
                         }

            return metric_dict, pred_dict




def main(args):

    if args.wandb:
        wandb.init(project=args.wandb_project, id=args.wbr, config=args, resume=None)

    captioner = UniqueCaptioner(args)
    keys = captioner.get_all_set_keys()
    keys.sort()

    if args.wbr is not None:
        fn = args.wbr
    else:
        fn = "output"
    fn = osp.join(args.output_dir, fn)
    fpkl = open(f"{fn}.pkl", 'wb')
    fpreds = open(f"{fn}_preds.pkl", 'wb')
    ftxt = open(f"{fn}.txt", 'w')

    metric_dict_total = {}

    log_dict = {}
    all_pred_dict = {}
    for i in range(args.n_sets_start, args.n_sets_start + args.n_sets):
        if i >= len(keys):
            break
        metric_dict, pred_dict = captioner.comb_run_set(keys[i])
        for k, v in metric_dict.items():
            value = metric_dict_total.get(k, torch.tensor(0.0))
            if isinstance(v, torch.Tensor):
                v = v.item()
            value += v
            metric_dict_total[k] = value
        log_dict[keys[i]] = metric_dict
        all_pred_dict[keys[i]] = pred_dict

        ftxt.write(f"{keys[i]}")
        for k, v in metric_dict.items():
            ftxt.write(f"{k}: {v},  ")
        ftxt.write("\n")

    for k, v in metric_dict_total.items():
        entry = v / min(args.n_sets, len(keys))
        entry = np.round(entry.tolist(), decimals=2) if isinstance(entry, torch.Tensor) else entry
        metric_dict_total[k] = entry

    pickle.dump(log_dict, fpkl)
    pickle.dump(metric_dict_total, fpkl)
    pickle.dump(all_pred_dict, fpreds)
    ftxt.write(str(metric_dict_total))

    fpkl.close()
    fpreds.close()
    ftxt.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser('lavila infer narrator', parents=[get_args_parser()])
    args = parser.parse_args()
    print(args)
    main(args)
