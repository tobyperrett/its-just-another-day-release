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
import numpy as np

import torch
import torchvision.transforms as transforms
import torchvision.transforms._transforms_video as transforms_video
from lavila.models.utils import inflate_positional_embeds

import torch.nn.functional as F

from lavila.data import datasets
from lavila.data.video_transforms import Permute
from lavila.models import models
from lavila.utils.preprocess import generate_tokenizer
from lavila.utils import distributed as dist_utils
from eval_narrator import decode_one
from lavila.models import prompt_predictor

import glob
import wandb


os.environ['WANDB_MODE'] = 'offline'
torch.set_printoptions(precision=2)
np.set_printoptions(precision=2)

def get_args_parser():
    parser = argparse.ArgumentParser(description='lavila train estimator from feature pkls', add_help=False)
    parser.add_argument('--dataset', default='ego', type=str, choices=['ego', 'tlm'])
    parser.add_argument('--root', default='storage/mp3_w1.0train_preds_deep.pkl', type=str, help='path to dataset root')
    parser.add_argument('--root_test', default='storage/mp3_w1.0train_preds.pkl', type=str, help='path to dataset root')
    parser.add_argument('--root_prompts', default='storage/prompts', type=str, help='path to dataset root')
    parser.add_argument('--prompts_sub_dir', default='clip_emb_narration', type=str, help='path to dataset root')
    parser.add_argument('--prompts_sub_dir_test', default='storage/prompts_test', type=str, help='path to dataset root')
    

    parser.add_argument('--output-dir', default='output', type=str, help='additional string for save checkpoint name')
    parser.add_argument('--output-str', default='arch10', type=str, help='additional string for save checkpoint name')
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--model', default='CSPairClsP', type=str)
    parser.add_argument('--loss', default='CSPairLoss', type=str)
    # System
    parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                        help='number of data loading workers per process')

    parser.add_argument('--wandb', default=0, help='Enable WandB logging', type=int)
    parser.add_argument('--wandb_project', default='predictor', type=str, help='wandb project name')

    parser.add_argument('--print-freq', default=1000, type=int, help='print frequency')
    parser.add_argument('--iters', default=999999999999, type=int)
    parser.add_argument('--val_iters', default=9999999, type=int)
    parser.add_argument('--val-freq', default=5000, type=int, help='val frequency')
    parser.add_argument('--test-freq', default=500, type=int, help='val frequency')
    parser.add_argument('--save-freq', default=5, type=int, help='val frequency')
    parser.add_argument('--epochs', default=31, type=int, help='val frequency')

    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--weight-decay', default=0.0, type=float)
    parser.add_argument('--project-embed-dim', default=256, type=int, help='embed dim after projection')
    parser.add_argument('--clip-length', default=4, type=int, help='clip length')
    parser.add_argument('--v_lambda', default=1.0, type=float)
    parser.add_argument('--t_lambda', default=1.0, type=float)
    parser.add_argument('--n_img_q', default=32, type=int, help="num_img_queries in narrator")

    parser.add_argument('--sch_gamma', default=0.1, type=float)
    parser.add_argument('--schedule_step_epochs', default=[10, 20], nargs='+')
    parser.add_argument('--un_rand_epoch', default=0, type=int, help="by default, randomise sets within batches for more variation. after this epoch, don't randomise")

    parser.add_argument('--n_prompts', default=1, type=int, help="num prompts")
    parser.add_argument('--n_dataset_prompts', default=10, type=int, help="num prompts")
    parser.add_argument('--set_size', default=10, type=int, help="num prompts")
    parser.add_argument('--set_rand_sampling', default=0, type=int, help="num prompts")
    parser.add_argument('--d_model', default=256, type=int, help="num prompts")
    parser.add_argument('--d_ff', default=1024, type=int, help="num prompts")
    parser.add_argument('--pp_feat_str', default="vis_feats", type=str, help="num prompts")
    parser.add_argument('--pp_nh', default=4, type=int, help="num prompts")
    parser.add_argument('--pp_nl', default=2, type=int, help="num prompts")




    return parser

LOG_KEYS = ["loss", "u_acc", "pred_unique", "pred_not_unique", "gt_unique", "pred_hist", "gt_hist"]

class TLMFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, args, mode="train"):
        self.args = args

        if mode == "test":
            self.all_vf = []
            self.all_tf = []
        else:
            vfn = osp.join(args.root, "CMD_5s_5fps_ViT-L-14_per_set.pth")
            tfn = osp.join(args.root, "CMD_text_feature_ViT-L-14_per_set.pth")

            self.all_vf = torch.load(vfn)
            self.all_tf = torch.load(tfn)


        self.sample_keys = [i for i in range(len(self.all_vf))]

        val_keys = self.sample_keys[::100]


        if mode == "train":
            self.key_list = [key for key in self.sample_keys if key not in val_keys]
        elif mode == "val":
            self.key_list = val_keys
        elif mode == "test":
            self.key_list = self.sample_keys
        self.mode = mode


    
    def __getitem__(self, index, index_is_key=False):
        key = index
        return_dict = {"set_key": key}

        vf = self.all_vf[key]['visual']
        tf = self.all_tf[key]['textual']

        # vf = torch.mean(vf[:,[0,5,10,20],:], dim=-2)
        vf = vf[:,13,:]

        this_set_size = vf.shape[0]
        if this_set_size < self.args.set_size:
            to_add = self.args.set_size - this_set_size
            vf = torch.cat([vf, vf[:to_add]], dim=0)
            tf = torch.cat([tf, tf[:to_add]], dim=0)
        
        if this_set_size > self.args.set_size:
            vf = vf[:self.args.set_size]
            tf = tf[:self.args.set_size]

        return_dict["vis_feats"] = vf
        # return_dict["target"] = 0

        return_dict["target_text_feats"] = tf

        return return_dict
    
    def get_sample_info(self, index, index_is_key=False):
        key = index
      
        return None

    def __len__(self):
        return len(self.key_list)



class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, args, mode="train"):
        self.args = args

        if mode == "test":
            with open(args.root_test, 'rb') as f:
                self.sample_infos = pickle.load(f)
        else:
            with open(args.root, 'rb') as f:
                self.sample_infos = pickle.load(f)
        self.sample_keys = list(self.sample_infos.keys())
        self.sample_keys.sort()

        val_keys = self.sample_keys[::100]


        if mode == "train":
            self.key_list = [key for key in self.sample_keys if key not in val_keys]
        elif mode == "val":
            self.key_list = val_keys
        elif mode == "test":
            self.key_list = self.sample_keys
        self.mode = mode

        if mode != "test":
            psd = args.prompts_sub_dir
        else:
            psd = args.prompts_sub_dir_test
        self.prompt_clip_embs = {}
        for i in range(self.args.n_dataset_prompts):
            with open(osp.join(self.args.root_prompts, f"p{i}", psd, "train_preds.pkl"), 'rb') as f:
                self.prompt_clip_embs[i] = pickle.load(f)

    def process_sample_info(self, sample_info, key):
        if type(sample_info) == dict:
            sample_info["key"] = key
            sample_info.pop(self.pop_key)

        elif type(sample_info) == list:
            sample_info = sample_info[0]
        return sample_info
    
    def load_text_embs(self, key):
        all_text_embs = []
        for i in range(self.args.n_dataset_prompts):
            all_text_embs.append(torch.tensor(self.prompt_clip_embs[i][key]["clip_text_emb"]))
        all_text_embs = torch.stack(all_text_embs)
        return all_text_embs   
    
    def __getitem__(self, index, index_is_key=False):
        if index_is_key:
            key = index
        else:
            key = self.key_list[index]
        return_dict = {"set_key": key}
        set_sample_infos = self.sample_infos[key]

        return_dict["vis_feats"] = torch.stack([s["vis_emb"] for k, s in set_sample_infos.items()])
        return_dict["enc_vis_feats"] = torch.stack([s["enc_vis_feats"] for k, s in set_sample_infos.items()])
        return_dict["target"] = torch.tensor([s["prompt_ids"][:self.args.n_prompts] for k, s in set_sample_infos.items()])

        return_dict["target_text_feats"] = torch.stack([self.load_text_embs(k) for k, s in set_sample_infos.items()])

        return return_dict
    
    def get_sample_info(self, index, index_is_key=False):
        if index_is_key:
            key = index
        else:
            key = self.key_list[index]        
    
        set_sample_infos = self.sample_infos[key]
        return set_sample_infos

    def __len__(self):
        return len(self.key_list)




def main(args):
    print(args)

    if args.wandb:
        wandb_id = args.output_str
        wandb.init(project=args.wandb_project, id=wandb_id, config=args, resume='allow')

    model = getattr(prompt_predictor,args.model)(args)
    loss_fn = getattr(prompt_predictor,args.loss)(args)
    model.cuda()
    # torch.backends.cudnn.benchmark = True
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule_step_epochs, gamma=args.sch_gamma)

    if args.dataset == "ego":
        val_dataset = FeatureDataset(args, mode="val")
        train_dataset = FeatureDataset(args, mode="train")
        test_dataset = FeatureDataset(args, mode="test")
    elif args.dataset == "tlm":
        val_dataset = TLMFeatureDataset(args, mode="val")
        train_dataset = TLMFeatureDataset(args, mode="train")
        test_dataset = TLMFeatureDataset(args, mode="test")       

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers, pin_memory=False, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers, pin_memory=False, drop_last=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers, pin_memory=False, drop_last=False
    )

    print('len(train_loader) = {}'.format(len(train_loader)))
    print('len(val_loader) = {}'.format(len(val_loader)))

    train_loss = {k: 0.0 for k in LOG_KEYS}


    for epoch in range(args.epochs):
    # if False:


        for data_iter, data in enumerate(train_loader):

            if data_iter >= args.iters:
                break

            data_dict = {k: v.cuda() for k, v in data.items() if type(v) == torch.Tensor}
            # data_dict = {k: v.cuda() for k, v in data.items()}
            # data_dict = create_input_and_target(data_dict, model)



            if epoch < args.un_rand_epoch:
                bb, vv, pp, dd = data_dict["target_text_feats"].shape
                vis_feats = []
                target_text_feats = []
                rand_elem_idxs = [torch.randperm(bb).t() for _ in range(vv)]
                rand_elem_idxs = torch.stack(rand_elem_idxs, dim=-1)
                for iv in range(vv):
                    idx = rand_elem_idxs[:, iv]
                    vis_feats.append(data_dict["vis_feats"][idx, iv, :])
                    target_text_feats.append(data_dict["target_text_feats"][idx, iv, :, :])
                data_dict["vis_feats"] = torch.stack(vis_feats, dim=1)
                data_dict["target_text_feats"] = torch.stack(target_text_feats, dim=1)

            if args.set_rand_sampling:
                n_idxs = np.random.randint(2, args.set_size+1)
                idxs = np.random.choice(args.set_size, n_idxs, replace=False)
                data_dict["vis_feats"] = data_dict["vis_feats"][:, idxs, :]
                # data_dict["target"] = data_dict["target"][:, idxs, :]
                data_dict["target_text_feats"] = data_dict["target_text_feats"][:, idxs, :, :]

            data_dict = model(data_dict)

            loss_dict = loss_fn(data_dict)
            loss_dict["loss"].backward()

            for k in train_loss.keys():
                train_loss[k] += loss_dict[k]

            torch.nn.utils.clip_grad_value_(parameters=model.parameters(), clip_value=1.0)

            optimizer.step()
            optimizer.zero_grad()

            if data_iter % args.print_freq == 0:
                log_dict = {k: loss_dict[k].tolist() for k in LOG_KEYS}
                print(f'epoch {epoch}, iter {data_iter}, {log_dict}')
                if args.wandb:
                    wandb.log(log_dict)
                train_loss = {k: 0.0 for k in LOG_KEYS}


            if data_iter % args.val_freq == 0:# and data_iter > 0:
                model.eval()
                with torch.no_grad():
                    val_loss = {k: 0.0 for k in LOG_KEYS}
                    all_diffs = []
                    for val_iter, val_data in enumerate(val_loader):
                        if val_iter >= args.val_iters:
                            break
                        data_dict = {k: v.cuda() for k, v in val_data.items() if type(v) == torch.Tensor}
                        # data_dict = create_input_and_target(data_dict, model)
                        output_dict = model(data_dict)
                        loss_dict = loss_fn(output_dict)
                        for k in val_loss.keys():
                            val_loss[k] += loss_dict[k]
                        all_diffs.append(loss_dict["diff"])

                    for k in val_loss.keys():
                        val_loss[k] /= len(val_loader)

                    log_dict = {k: val_loss[k].tolist() for k in LOG_KEYS}
                    print(f'VAL epoch {epoch}, iter {data_iter}, {log_dict}')
                    # log_dict["diff"] = all_diffs

                    out_fn = f"{osp.join(args.output_dir, args.output_str)}_{args.model}_{args.loss}_p{args.n_dataset_prompts}_e{epoch}"
                    with open(f"{out_fn}_diffs.pkl", 'wb') as f:
                        pickle.dump(all_diffs, f)
                    if args.wandb:
                        wandb.log({f'val_{k}': v for k, v in val_loss.items()})
                model.train()            



        scheduler.step()



        if (epoch > 0) and (epoch % args.test_freq == 0):
            model.eval()
            with torch.no_grad():
                out_dict = {}

                for val_iter, val_data in enumerate(test_loader):

                    data_dict = {k: v.cuda() for k, v in val_data.items() if type(v) == torch.Tensor}
                    output_dict = model(data_dict)
                    loss_dict = loss_fn(output_dict)
                    
                    preds = loss_dict["prompt_id_preds"]
                    gt = loss_dict["prompt_id_gt"]
                    keys = val_data["set_key"]

                    # print(preds.shape, gt.shape)
                    # exit(0)``

                    for i in range(preds.shape[0]):
                        set_sample_info = test_dataset.get_sample_info(keys[i], index_is_key=True)
                        set_sample_info_keys = list(set_sample_info.keys())
                        for j in range(preds.shape[1]):
                            set_sample_info[set_sample_info_keys[j]]["prompt_id_preds"] = preds[i][j].tolist()
                            set_sample_info[set_sample_info_keys[j]]["prompt_id_gt"] = gt[i][j].tolist()
                        out_dict[keys[i]] = set_sample_info

                out_fn = f"{osp.join(args.output_dir, args.output_str)}_{args.model}_{args.loss}_p{args.n_dataset_prompts}_e{epoch}"
                with open(f"{out_fn}.pkl", 'wb') as f:
                    pickle.dump(out_dict, f)

            model.train()

        if epoch % args.save_freq == 0:
            out_fn = f"{osp.join(args.output_dir, args.output_str)}_{args.model}_p{args.n_dataset_prompts}_e{epoch}"
            with open(f"{out_fn}_model.pkl", 'wb') as f:
                save_dict = {"state_dict": model.state_dict(), "args": args}
                torch.save(save_dict, f)




    



if __name__ == '__main__':
    parser = argparse.ArgumentParser('lavila infer narrator', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
