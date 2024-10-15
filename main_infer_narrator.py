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

from lavila.data import datasets
from lavila.data.video_transforms import Permute
from lavila.models import models
from lavila.utils.preprocess import generate_tokenizer
from lavila.utils import distributed as dist_utils
from eval_narrator import decode_one


class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        return index, self.dataset[index]

    def __len__(self):
        return len(self.dataset)


def get_args_parser():
    parser = argparse.ArgumentParser(description='lavila infer narrator', add_help=False)
    parser.add_argument('--dataset', default='ego4d', type=str, choices=['ego4d'])
    parser.add_argument('--root', default='/jmain02/home/J2AD001/wwp01/txp48-wwp01/ego4d_data/v2/video_256_30fps_300s',
                        type=str, help='path to dataset root')
    parser.add_argument('--chunk_len', default=300, type=int, help='chunk length of mp4s')
    parser.add_argument('--metadata', default='/jmain02/home/J2AD001/wwp01/txp48-wwp01/lavila_input/val_correct_incorrect_lavila_h5-5.pkl',
                        type=str, help='path to metadata file')
    parser.add_argument('--metadata_sets', default=None, type=str, help='pkl file containing sets of similar videos')
    parser.add_argument('--output-dir', default='./', type=str, help='output dir')
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--use-half', action='store_true')
    parser.add_argument('--clip-length', default=4, type=int, help='clip length')
    parser.add_argument('--clip-stride', default=16, type=int, help='clip stride')
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint')
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

    parser.add_argument('--return_original_narration', default=1, type=int)
    parser.add_argument('--return_positive', default=0, type=int)
    parser.add_argument('--return_negative', default=0, type=int)
    parser.add_argument('--before_after_choice', default="random",choices=['random', 'before', 'after'], type=str, help="return before, after, or random")
    parser.add_argument('--dataloader_text', default="narration", choices=["caption_lav-base", "narration"], help='which type of text to load from dataset')

    parser.add_argument('--iters', default=999999999999, type=int)
    parser.add_argument('--extract_vis_feats', default=1, type=int, help='extract output of visual model')
    parser.add_argument('--cls_at_last', default=0, type=int, help='extract cls representation used in contrastive training, rather than the 16x16 grid')
    parser.add_argument('--extract_bos', default=0, type=int, help='extract bos token after one VCLM pass')
    parser.add_argument('--extract_captions', default=0, type=int, help="extract generated captions")
    parser.add_argument('--extract_prompt_feats', default=0, type=int, help='extract feats after gt single narrarion')
    parser.add_argument('--merge_results', action='store_true')

    # options for loading multiple similar videos
    parser.add_argument('--load_similar_videos', default=0, type=int)
    parser.add_argument('--load_similar_videos_range', default=[2], help="one int for constant, two ints for range")



    #override variable clip length
    parser.add_argument('--fixed_clip_len', default=0, type=float, help="fixed clip_length")
    parser.add_argument('--prepend', default=None, choices=[None, "cur_verb"], help="Prepend a verb to the narration")
    parser.add_argument('--img_q_drop', type=float, default=0.0, help='chance between 0 and 1 to drop image queries')

    # options for estimator network
    parser.add_argument('--estimator', default=None, type=str, help="estimator network")
    parser.add_argument('--sim_type', type=str, nargs='+', help="temporal-vid, same-caption-vid, same-caption-ds, caption-embed-vid, same-narration-vid, same-narration-ds, narration-embed-vid, visual-embed-vid, rand")
    parser.add_argument('--uct_type', default=None, choices=[None, 'encoder', 'attention', 'residual', 'shared-residual', 'weighted', 'none-debug'], type=str, help="type of uct to use")
    parser.add_argument('--uct_heads', default=1, type=int)
    parser.add_argument('--uct_layers', default=1, type=int)
    parser.add_argument('--uct_dim_feedforward', default=2048, type=int)
    parser.add_argument('--img_q_pe', default=0, type=int, help="Add positional encodings to img queries in estimator")
    parser.add_argument('--n_img_q', default=256, type=int, help="num_img_queries in narrator")

    parser.add_argument('--prompt', default=None, type=str)
    
    return parser


def main(args):
    dist_utils.init_distributed_mode(args)
    print(args)
    if args.extract_captions or args.extract_bos:
        assert args.cls_at_last == 0


    if args.resume:
        ckpt_path = args.resume
    elif osp.isfile(osp.join(args.output_dir, 'checkpoint_best.pt')):
        ckpt_path = osp.join(args.output_dir, 'checkpoint_best.pt')
    else:
        raise Exception('no checkpoint found')

    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = OrderedDict()
    for k, v in ckpt['state_dict'].items():
        state_dict[k.replace('module.', '')] = v


    # create model
    old_args = ckpt['args']
    print('=> creating model: {}'.format(old_args.model))
    tokenizer = generate_tokenizer(old_args.model)
    model = getattr(models, old_args.model)(
        text_use_cls_token=old_args.use_cls_token,
        gated_xattn=old_args.gated_xattn if hasattr(old_args, 'gated_xattn') else None,
        timesformer_gated_xattn=old_args.timesformer_gated_xattn if hasattr(old_args, 'timesformer_gated_xattn') else None,
        num_frames=old_args.clip_length,
        drop_path_rate=0,
        tokenizer=tokenizer,
        args=args
    )
    model.cuda()
    model.load_state_dict(state_dict, strict=False)
    # model.load_state_dict(state_dict, strict=True)
    print("=> loaded resume checkpoint '{}' (epoch {})".format(args.resume, ckpt['epoch']))

    # torch.backends.cudnn.benchmark = True

    # Data loading
    print("=> creating dataset")

    crop_size = 224 if '336PX' not in old_args.model else 336
    val_transform = transforms.Compose([
        Permute([3, 0, 1, 2]),  # T H W C -> C T H W
        transforms.Resize(crop_size),
        transforms.CenterCrop(crop_size),
        (transforms_video.NormalizeVideo(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]) if 'OPENAI' not in old_args.model else
            transforms_video.NormalizeVideo(mean=[108.3272985, 116.7460125, 104.09373615000001], std=[68.5005327, 66.6321579, 70.32316305])),
    ])

    val_dataset = datasets.VideoCaptionDatasetCLIP(
        args.dataset,
        args.root,
        args.metadata,
        transform=val_transform,
        is_training=False,
        tokenizer=tokenizer,
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
    val_dataset = IndexedDataset(val_dataset)

    print(len(val_dataset))

    if args.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=True)
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
            inputs = [inputs_and_info["frames"], inputs_and_info["text"], inputs_and_info["relevancy"]]
            sample_keys = inputs_and_info["keys"]

            if data_iter >= args.iters:
                break
            indices = indices.tolist()
            if data_iter % args.print_freq == 0:
                print("finished {}/{} in {}".format(data_iter, len(val_loader), time.time() - end))
                end = time.time()
            if len(inputs) == 2 or len(inputs) == 3:
                images = inputs[0].cuda(non_blocking=True)
                if args.use_half:
                    images = images.half()

                if "CLIP" in old_args.model:
                    text = inputs[1].cuda(non_blocking=True)
                    image_features, image_deep_features = dist_utils.get_model(model).encode_image(images, return_features=True)
                    text_features = dist_utils.get_model(model).encode_text(text)

                    np_image_features = image_features.cpu().numpy()
                    np_image_deep_features = image_deep_features.cpu().numpy()
                    np_text_features = text_features.cpu().numpy()
                    for j in range(image_features.shape[0]):
                        sample_info = val_loader.dataset.dataset.get_sample_metadata(indices[j])
                        sample_info["clip_vis_feats"] = np_image_features[j]
                        sample_info["enc_vis_feats"] = np_image_deep_features[j]
                        sample_info["clip_text_emb"] = np_text_features[j]
                        fn = osp.join(args.output_dir, "clip_emb_{}".format(args.dataloader_text), '{}.pkl'.format(sample_info["key"]))
                        pickle.dump(sample_info, open(fn, 'wb'))
                    continue

                if args.cls_at_last:
                    image_features = dist_utils.get_model(model).visual(images)
                    folder_suffix = "_proj"
                else:
                    image_features, _ = dist_utils.get_model(model).encode_image(images)
                    folder_suffix = ""

                
                if args.img_q_drop > 0:
                    folder_suffix += f"_qd{args.img_q_drop}"
                    image_features = image_features * (torch.rand(image_features.shape, device=image_features.device) > args.img_q_drop).float()


                if not isinstance(image_features, (list, tuple)):
                    image_tokens = image_features
                else:
                    image_tokens = image_features[1]

                if args.extract_vis_feats:
                    np_image_tokens = image_tokens.cpu().numpy()
                    for j in range(image_tokens.shape[0]):
                        sample_info = val_loader.dataset.dataset.get_sample_metadata(indices[j])
                        sample_info["vis_feats"] = np_image_tokens[j]
                        fn = osp.join(args.output_dir, "vis_feats"+folder_suffix, '{}.pkl'.format(sample_info["key"]))
                        pickle.dump(sample_info, open(fn, 'wb'))

                if args.extract_bos:
                    bos_tokens = dist_utils.get_model(model).generate_feats(image_tokens, tokenizer, prompt=None)
                    np_bos_tokens = bos_tokens.cpu().numpy()
                    for j in range(bos_tokens.shape[0]):
                        sample_info = val_loader.dataset.dataset.get_sample_metadata(indices[j])
                        sample_info["bos_feats"] = np_bos_tokens[j]
                        fn = osp.join(args.output_dir, "bos_feats", '{}.pkl'.format(sample_info["key"]))
                        pickle.dump(sample_info, open(fn, 'wb'))

                if args.extract_prompt_feats:
                    prompts = [val_loader.dataset.dataset.get_sample_metadata(indices[j])["narration"] for j in range(len(indices))]
                    bos_tokens = dist_utils.get_model(model).generate_feats(image_tokens, tokenizer, prompt=prompts)
                    np_bos_tokens = bos_tokens.cpu().numpy()
                    for j in range(bos_tokens.shape[0]):
                        sample_info = val_loader.dataset.dataset.get_sample_metadata(indices[j])
                        sample_info["prompt_feats"] = np_bos_tokens[j]
                        fn = osp.join(args.output_dir, "prompt_feats", '{}.pkl'.format(sample_info["key"]))
                        pickle.dump(sample_info, open(fn, 'wb'))



                if args.extract_captions:
                    if args.load_similar_videos:
                        all_sample_infos = {}

                        token_list = [(image_tokens, "orig")]

                        if hasattr(dist_utils.get_model(model), "uct"):
                            image_tokens_u = dist_utils.get_model(model).uct(image_tokens)
                            token_list.append((image_tokens_u, "unique"))
                        
                        for i_tok, i_t_k in token_list:
                            sample_infos_per_sim = []
                            for sim_idx in range(image_tokens.shape[-3]):
                                sample_infos = do_caption(args, model, i_tok[:,sim_idx,:,:], tokenizer, sample_keys[sim_idx], val_loader)
                                sample_infos_per_sim.append(sample_infos)
                            all_sample_infos[i_t_k] = list(zip(*sample_infos_per_sim))

                        for i in range(len(all_sample_infos["orig"])):
                            sample_info = {"orig": all_sample_infos["orig"][i]}
                            if "unique" in all_sample_infos:
                                sample_info["unique"] = all_sample_infos["unique"][i]
                            
                            fn = osp.join(args.output_dir, "captions", '{}.pkl'.format(sample_info["orig"][0]["set_key"]))
                            pickle.dump([sample_info], open(fn, 'wb'))


                    else:
                        all_sample_infos = do_caption(args, model, image_tokens, tokenizer, sample_keys, val_loader)
                        for sample_info in all_sample_infos:
                            fn = osp.join(args.output_dir, "captions", '{}.pkl'.format(sample_info["key"]))
                            pickle.dump([sample_info], open(fn, 'wb'))
                    # id_offset += generated_text_ids.shape[0]

    pickle.dump(all_captions_cache, open(osp.join(args.output_dir, 'cache.{}.pkl'.format(args.rank)), 'wb'))

    if args.merge_results:
        torch.distributed.barrier()
        disorded_list = []
        total_num = 0
        if args.rank == 0:
            for i in range(args.world_size):
                print('=> reading {}'.format(osp.join(args.output_dir, f'cache.{i}.pkl')))
                sublist = pickle.load(open(osp.join(args.output_dir, f'cache.{i}.pkl'), 'rb'))
                disorded_list.append(sublist)
                total_num += len(sublist)
            ordered_list = []
            for i in range(total_num):
                ordered_list.append(disorded_list[i % args.world_size][i // args.world_size])
            print(f"{len(val_dataset)}/{len(ordered_list)}")
            ordered_list = ordered_list[:len(val_dataset)]
            pickle.dump(ordered_list, open(osp.join(args.output_dir, 'total.pkl'), 'wb'))
            for i in range(args.world_size):
                print('=> deleting {}'.format(osp.join(args.output_dir, f'cache.{i}.pkl')))
                os.remove(osp.join(args.output_dir, f'cache.{i}.pkl'))

def do_caption(args, model, image_tokens, tokenizer, sample_keys, val_loader, target=None, return_text_only=False, prompt=None):
    if args.caption_sample == 'multinomial_sample':
        generated_text_ids, ppls = dist_utils.get_model(model).generate(
            image_tokens,
            tokenizer,
            target=None,
            max_text_length=args.caption_max_len,
            top_k=args.caption_top_k,
            top_p=args.caption_top_p,
            num_return_sequences=args.caption_num_return_sequences,
            temperature=args.caption_temperature,
            early_stopping=args.caption_early_stop,
            arg_prompt=prompt
        )
    elif args.caption_sample == 'beam_sample':
        generated_text_ids, ppls = dist_utils.get_model(model).beam_sample(
            image_tokens,
            tokenizer,
            target=None,
            max_text_length=args.caption_max_len,
            top_k=args.caption_top_k,
            top_p=args.caption_top_p,
            temperature=args.caption_temperature,
            length_penalty=args.caption_length_penalty,
            num_beams=args.caption_num_beams,
            num_return_sequences=args.caption_num_return_sequences,
        )
    elif args.caption_sample == 'group_beam_search':
        assert args.caption_num_beam_groups > 1 and args.caption_num_beams % args.caption_num_beam_groups == 0
        generated_text_ids, ppls = dist_utils.get_model(model).group_beam_search(
            image_tokens,
            tokenizer,
            target=None,
            max_text_length=args.caption_max_len,
            top_k=args.caption_top_k,
            top_p=args.caption_top_p,
            temperature=args.caption_temperature,
            length_penalty=args.caption_length_penalty,
            num_beams=args.caption_num_beams,
            num_beam_groups=args.caption_num_beam_groups,
            num_return_sequences=args.caption_num_return_sequences,
        )


    all_sample_infos = []

    for j in range(generated_text_ids.shape[0] // args.caption_num_return_sequences):
        generated_text_str_list = []
        ppls_list = []
        for k in range(args.caption_num_return_sequences):
            jj = j * args.caption_num_return_sequences + k
            generated_text_str = decode_one(generated_text_ids[jj], tokenizer)
            generated_text_str_list.append(generated_text_str)
            ppls_list.append(ppls[jj].item())
        
        if args.caption_num_return_sequences == 1:
            save_text = generated_text_str
            save_ppl = ppls[jj].item()
            # if args.merge_results:
            #     all_captions_cache.append((video_uid, t_start, t_end, generated_text_str, ppls[jj].item()))
        else:
            save_text = generated_text_str_list
            save_ppl = ppls_list
            # if args.merge_results:
            #     all_captions_cache.append((video_uid, t_start, t_end, generated_text_str_list, ppls_list))

        if return_text_only:
            all_sample_infos.append(save_text)
        else:
            sample_info = val_loader.dataset.dataset.get_sample_metadata(int(sample_keys[j]), i_is_key=True)

            sample_info["caption"] = save_text
            sample_info["ppls"] = save_ppl
            all_sample_infos.append(sample_info)
    return all_sample_infos

if __name__ == '__main__':
    parser = argparse.ArgumentParser('lavila infer narrator', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
