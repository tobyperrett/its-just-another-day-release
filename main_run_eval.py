import os
import os.path as osp
import sys
import subprocess
import random
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_name" )
    # parser.add_argument("--runs_dir", default="/jmain02/home/J2AD001/wwp01/txp48-wwp01/lavila_output")
    parser.add_argument("--runs_dir", default="/jmain02/home/J2AD001/wwp01/txp48-wwp01/lavila_output/GPT2-TF5-8iq-5s-v2/narrator")
    parser.add_argument("--results_dir", default="/jmain02/home/J2AD001/wwp01/txp48-wwp01/lavila_output/recall_dev")
    parser.add_argument("--add_dir", default=None)
    parser.add_argument("--metadata_sets_dir", default="/jmain02/home/J2AD001/wwp01/txp48-wwp01/lavila_input")
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--set_size", type=int, default=3)
    parser.add_argument("--infer_model", default='CLIP_OPENAI_TIMESFORMER_BASE')
    parser.add_argument("--evl_resume", default='/jmain02/home/J2AD001/wwp01/txp48-wwp01/lavila_ckpts/clip_openai_timesformer_base.narrator_rephraser.ep_0005.md5sum_d73a9c.pth')
    parser.add_argument("--evl_model", default='CLIP_OPENAI_TIMESFORMER_BASE')
    parser.add_argument("--ckpt_str", default='_0003')
    parser.add_argument("--sets", default='val_recall')
    parser.add_argument("--extract", default=1, type=int)
    parser.add_argument("--merge", default=1, type=int)
    parser.add_argument("--evaluate", default=1, type=int)
    parser.add_argument('--val_caption_set', choices=["unique", "orig"], default='unique', type=str, help='unique: use captions after U, orig: captions before U ()i.e. base model')
    parser.add_argument('--fixed_clip_len', default=0, type=float, help="fixed clip_length")
    parser.add_argument("--prompt", default=None, type=str)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gpus", type=int, default=2)

    parser.add_argument("--add_args", nargs=argparse.REMAINDER,
                         default=None
                        )
    parser.add_argument("--debug", default=0, type=int)

    args = parser.parse_args()

    run_folder = osp.join(args.runs_dir, args.run_name)

    # caption_pickle_folder = osp.join(run_folder, f'evl_recall_{args.set_size}', 'captions')

    run_out_folder_name = f'{args.sets.replace("val", "evl")}_{args.set_size}_fcl{args.fixed_clip_len}'

    if args.add_dir is not None:
        run_out_folder_name = osp.join(args.add_dir, run_out_folder_name)

    caption_pickle_folder = osp.join(run_folder, run_out_folder_name, 'captions')

    if args.infer_model == "CLIP_OPENAI_TIMESFORMER_BASE":
        infer_root = '/jmain02/home/J2AD001/wwp01/txp48-wwp01/ego4d_data/v2/video_256_30fps_300s'
    elif args.infer_model == "CLIP_OPENAI_TIMESFORMER_LARGE_336PX_DISTILBERT_BASE":
        infer_root = '/jmain02/home/J2AD001/wwp01/txp48-wwp01/ego4d_data/v2/video_384_30fps_300s'  


    if args.metadata is None:
        metadata = osp.join(args.metadata_sets_dir, f"{args.sets}.pkl")
        metadata_sets = osp.join(args.metadata_sets_dir, f"{args.sets}_sets.pkl")
    else:
        metadata = osp.join(args.metadata_sets_dir, f"{args.metadata}.pkl")
        metadata_sets = osp.join(args.metadata_sets_dir, f"{args.sets}.pkl")

    if args.extract:
        # Make folders
        os.makedirs(caption_pickle_folder, exist_ok=True)

        # Extract
        extract_command = ['torchrun',
            '--rdzv_backend=c10d',
            f'--rdzv_endpoint=localhost:{random.randint(10000, 19999)}',
            f'--nproc_per_node={args.gpus}',
            'main_infer_narrator.py',
            f'--resume={osp.join(run_folder, f"checkpoint{args.ckpt_str}.pt")}',
            f'--output-dir={osp.join(run_folder, run_out_folder_name)}',
            '--return_positive=0', '--return_negative=0',
            f'--metadata={metadata}',
            f'--metadata_sets={metadata_sets}',
            '--extract_captions=1', '--extract_vis_feats=0', '--extract_bos=0', '--cls_at_last=0',
            '--load_similar_videos=1', f'--load_similar_videos_range={args.set_size-1}', '--sim_type=rand',
            f'--fixed_clip_len={args.fixed_clip_len}',
            f'--root={infer_root}',
            f'--batch-size={args.batch_size}'
        ]

        if args.prompt is not None:
            extract_command.append(f'--prompt="{args.prompt}"')

        if args.add_args is not None:
            if type(args.add_args) == list:
                args.add_args = " ".join(args.add_args)
            print(args.add_args)
            extract_command.extend([s for s in args.add_args.split()])

        for c in extract_command:
            print(c)
        # print(extract_command)
        # exit(0)
        if (not args.debug):
            subprocess.run(extract_command)

    if args.merge:
        # Merge preds
        merge_command = [
            'python', 'scripts/merge_pkl_for_recall.py',
            '--directory', caption_pickle_folder,
            '--val_caption_set', args.val_caption_set,
        ]
        print(merge_command)
        if not args.debug:
            subprocess.run(merge_command)

    if args.evaluate:
        # Eval lav
        set_fn = f"{args.sets.replace('val', 'train')}_sets.pkl"
        log_fn = f"{args.run_name}{args.ckpt_str}_fcl{args.fixed_clip_len}_{args.evl_model}_{args.set_size}_{args.sets}"
        if args.add_dir is not None:
            log_fn = log_fn + f'_{args.add_dir}'



        if args.evl_model == "CLIP_OPENAI_TIMESFORMER_BASE":
            evl_resume = "/jmain02/home/J2AD001/wwp01/txp48-wwp01/lavila_ckpts/clip_openai_timesformer_base.narrator_rephraser.ep_0005.md5sum_d73a9c.pth"
            evl_root = '/jmain02/home/J2AD001/wwp01/txp48-wwp01/ego4d_data/v2/video_256_30fps_300s'
        elif args.evl_model == "CLIP_OPENAI_TIMESFORMER_LARGE_336PX_DISTILBERT_BASE":
            evl_resume = "/jmain02/home/J2AD001/wwp01/txp48-wwp01/lavila_ckpts/clip_openai_timesformer_large_336px_distilbert_base.baseline.ep_0003.pth"
            evl_root = '/jmain02/home/J2AD001/wwp01/txp48-wwp01/ego4d_data/v2/video_384_30fps_300s'  
        elif args.evl_model == "CLIP_OPENAI_TIMESFORMER_LARGE":
            evl_root = '/jmain02/home/J2AD001/wwp01/txp48-wwp01/ego4d_data/v2/video_256_30fps_300s'
            evl_resume = "/jmain02/home/J2AD001/wwp01/txp48-wwp01/lavila_ckpts/clip_openai_timesformer_large.narrator_rephraser.ep_0003.md5sum_c89337.pth"
        elif args.evl_model == "CLIP_OPENAI_VITL14_336PX":
            evl_resume = ""
            evl_root = '/jmain02/home/J2AD001/wwp01/txp48-wwp01/ego4d_data/v2/video_384_30fps_300s'        
        else:
            raise NotImplementedError
        
        evl_command = [
            'torchrun',
            '--rdzv_backend=c10d',
            f'--rdzv_endpoint=localhost:{random.randint(10000, 19999)}',
            '--nproc_per_node=1',
            'main_pretrain_context.py',
            f'--root={evl_root}',
            f'--resume={evl_resume}',
            f'--output-dir={args.results_dir}',
            f'--model={args.evl_model}', '--norm-embed', '--freeze-temperature', '--fix-lr',
            '--contrastive-use-vissl', '--use-checkpoint',
            '--return_positive=0', '--return_negative=0', '--return_original_narration=1', '--batch-size=8',
            '--assignment=1',
            f'--metadata={osp.join(caption_pickle_folder, "train_preds.pkl")}',
            f'--metadata_sets={osp.join(args.metadata_sets_dir, set_fn)}',
            f'--log_fn={log_fn}', '--val_recall_only', '--load_similar_videos=1',
            f'--load_similar_videos_range={args.set_size - 1}', '--sim_type=rand',
            f'--fixed_clip_len={args.fixed_clip_len}',
            f'--preprocess_text=first-sentence-only',
        ]
        print(evl_command)
        for c in evl_command:
            print(c)
        if not args.debug:
            subprocess.run(evl_command)


if __name__ == "__main__":
    main()
