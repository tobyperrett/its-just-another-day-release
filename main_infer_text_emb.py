# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import argparse
from collections import OrderedDict
import os
import os.path as osp
import pickle
import time

import torch
import torchvision.transforms as transforms
import torchvision.transforms._transforms_video as transforms_video
from lavila.models.utils import inflate_positional_embeds

from lavila.data import datasets
from lavila.data.video_transforms import Permute
from lavila.models import models
from lavila.utils.preprocess import generate_tokenizer
from lavila.utils import distributed as dist_utils
from eval_narrator import decode_one

import glob

def get_args_parser():
    parser = argparse.ArgumentParser(description='lavila infer narrator', add_help=False)
    parser.add_argument('--dataset', default='ego4d', type=str, choices=['ego4d'])
    parser.add_argument('--root', default='/jmain02/home/J2AD001/wwp01/txp48-wwp01/lavila_output/infer_train_3000/bos_feats',
                        type=str, help='path to dataset root')
    parser.add_argument('--output-dir', default='/jmain02/home/J2AD001/wwp01/txp48-wwp01/lavila_output/infer_train_3000', type=str, help='output dir')
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--use-half', action='store_true')
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint')
    parser.add_argument('--model', default='CLIP_OPENAI_TIMESFORMER_BASE', type=str)
    parser.add_argument('--clip-length', default=4, type=int, help='clip length')
    # System
    parser.add_argument('--print-freq', default=100, type=int, help='print frequency')
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
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

    parser.add_argument('--iters', default=999999999999, type=int)
    parser.add_argument('--text_to_encode', choices=["narration", "caption"], default='narration', type=str)
    parser.add_argument('--merge_results', action='store_true')

    return parser



class TextDataset(torch.utils.data.Dataset):
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.in_memory = False

        if args.root.endswith(".pkl"):
            with open(args.root, 'rb') as f:
                self.files_dict = pickle.load(f)
                self.in_memory = True
                self.key_list = list(self.files_dict.keys())
        else:
            # read all pickle files in args.root with glob
            self.files_dict = {}
            for fn in glob.glob(osp.join(args.root, '*.pkl')):
                key = osp.basename(fn).split("_")[0].replace('.pkl', '')
                key = int(key)

                self.files_dict[key] = fn

            self.key_list = list(self.files_dict.keys())
            # get name of enclosing folder
            self.pop_key = self.args.root.split(osp.sep)[-1]

    def process_sample_info(self, sample_info, key):
        if type(sample_info) == dict:
            sample_info["key"] = key
            if not self.in_memory:
                sample_info.pop(self.pop_key)

        elif type(sample_info) == list:
            sample_info = sample_info[0]
        return sample_info

    def __getitem__(self, index):
        key = self.key_list[index]
        fn = self.files_dict[key]

        if self.in_memory:
            sample_info = self.files_dict[key]
        else:
            with open(fn, 'rb') as f:
                sample_info = pickle.load(f)

        sample_info = self.process_sample_info(sample_info, key)

        sample_info["key"] = key

        tokenized_text = self.tokenizer(sample_info[self.args.text_to_encode])

        return {"text_tokens": tokenized_text, "key": key}
    
    def get_sample_info(self, key):

        if self.in_memory:
            sample_info = self.files_dict[key]
        else:
            fn = self.files_dict[key]
            with open(fn, 'rb') as f:
                sample_info = pickle.load(f)
        sample_info = self.process_sample_info(sample_info, key)
        return sample_info
    
    def __len__(self):
        return len(self.files_dict)

class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        return index, self.dataset[index]

    def __len__(self):
        return len(self.dataset)


def main(args):
    dist_utils.init_distributed_mode(args)
    print(args)

    if args.resume:
        ckpt_path = args.resume
    elif osp.isfile(osp.join(args.output_dir, 'checkpoint_best.pt')):
        ckpt_path = osp.join(args.output_dir, 'checkpoint_best.pt')
    else:
        raise Exception('no checkpoint found')
    
    tokenizer = generate_tokenizer(args.model)
    

    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v

    old_args = ckpt['args']
    print('=> creating model: {}'.format(old_args.model))
    model = getattr(models, old_args.model)(
        text_use_cls_token=old_args.use_cls_token,
        project_embed_dim=old_args.project_embed_dim,
        gated_xattn=False if 'gated_xattn' not in old_args else old_args.gated_xattn,
        timesformer_gated_xattn=False if 'timesformer_gated_xattn' not in old_args else old_args.timesformer_gated_xattn,
        timesformer_freeze_space=False if 'timesformer_freeze_space' not in old_args else old_args.timesformer_freeze_space,
        freeze_lm_vclm=False if 'freeze_lm_vclm' not in old_args else old_args.freeze_lm_vclm,
        freeze_visual_vclm=False if 'freeze_visual_vclm' not in old_args else old_args.freeze_visual_vclm,
        num_frames=args.clip_length,
        drop_path_rate=0,
    )
    model.cuda()
    if 'TIMESFORMER' in old_args.model or 'EGOVLP' in old_args.model:
        # inflate weight
        print('=> inflating PE in models due to different frame numbers')
        state_dict = inflate_positional_embeds(
            model.state_dict(), state_dict,
            num_frames=args.clip_length,
            load_temporal_fix='bilinear',
        )
    model.load_state_dict(state_dict, strict=True)
    print("=> loaded resume checkpoint '{}' (epoch {}, best_metric = {})".format(args.resume, ckpt['epoch'], ckpt['best_acc1']))

    torch.backends.cudnn.benchmark = True

    # Data loading
    print("=> creating dataset")

    val_dataset = TextDataset(args, tokenizer)
    val_dataset = IndexedDataset(val_dataset)

    print(len(val_dataset))

    if args.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    else:
        val_sampler = None

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=False
    )
    print('len(val_loader) = {}'.format(len(val_loader)))

    model.eval()
    if args.use_half:
        model.half()

    id_offset = 0
    all_captions_cache = []
    end = time.time()
    with torch.no_grad():
        # for data_iter, (indices, inputs) in enumerate(val_loader):
        for data_iter, (indices, inputs_and_info) in enumerate(val_loader):
            inputs, sample_keys = [inputs_and_info["text_tokens"], inputs_and_info["key"]]

            if data_iter >= args.iters:
                break
            indices = indices.tolist()
            if data_iter % args.print_freq == 0:
                print("finished {}/{} in {}".format(data_iter, len(val_loader), time.time() - end))
                end = time.time()
            
            
            text = inputs.cuda(non_blocking=True)
            if args.use_half:
                text = text.half()

            text_features = dist_utils.get_model(model).encode_text(text)
            

            if not isinstance(text_features, (list, tuple)):
                text_tokens = text_features
            else:
                text_tokens = text_features[1]

            np_text_tokens = text_tokens.cpu().numpy()
            for j in range(text_tokens.shape[0]):
                sample_info = val_loader.dataset.dataset.get_sample_info(int(sample_keys[j]))
                emb_key = f"{args.text_to_encode}_emb"
                sample_info[emb_key] = np_text_tokens[j]
                fn = osp.join(args.output_dir, emb_key, '{}.pkl'.format(sample_info["key"]))
                pickle.dump(sample_info, open(fn, 'wb'))







    # pickle.dump(all_captions_cache, open(osp.join(args.output_dir, 'cache.{}.pkl'.format(args.rank)), 'wb'))

    # if args.merge_results:
    #     torch.distributed.barrier()
    #     disorded_list = []
    #     total_num = 0
    #     if args.rank == 0:
    #         for i in range(args.world_size):
    #             print('=> reading {}'.format(osp.join(args.output_dir, f'cache.{i}.pkl')))
    #             sublist = pickle.load(open(osp.join(args.output_dir, f'cache.{i}.pkl'), 'rb'))
    #             disorded_list.append(sublist)
    #             total_num += len(sublist)
    #         ordered_list = []
    #         for i in range(total_num):
    #             ordered_list.append(disorded_list[i % args.world_size][i // args.world_size])
    #         print(f"{len(val_dataset)}/{len(ordered_list)}")
    #         ordered_list = ordered_list[:len(val_dataset)]
    #         pickle.dump(ordered_list, open(osp.join(args.output_dir, 'total.pkl'), 'wb'))
    #         for i in range(args.world_size):
    #             print('=> deleting {}'.format(osp.join(args.output_dir, f'cache.{i}.pkl')))
    #             os.remove(osp.join(args.output_dir, f'cache.{i}.pkl'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('lavila infer narrator', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
