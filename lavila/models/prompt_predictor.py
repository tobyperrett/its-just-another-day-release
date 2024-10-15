import torch
from einops import rearrange
import torch.nn.functional as F
import math
import torch.nn as nn



class P1(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.d_model = 256
        self.d_feedforward = 512
        self.n_heads = 4
        self.n_layers = 2
        self.n_prompts = args.n_dataset_prompts
        self.n_videos = args.set_size

        # self.input_project = torch.nn.Linear(self.d_input, self.d_model)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self.d_model, dim_feedforward=self.d_feedforward, nhead=self.n_heads, batch_first=True)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)
        self.latent = torch.nn.Parameter(torch.zeros(1,1,self.d_model))

        self.head_norm = torch.nn.LayerNorm(self.d_model)
        self.cls_head = torch.nn.Linear(self.d_model, self.n_prompts*2)
        self.p_head = torch.nn.Linear(self.d_model, self.n_prompts)

        self.video_tag = torch.nn.Parameter(torch.zeros(1, self.n_videos, self.d_model))
        # self.video_tag = torch.nn.Parameter(torch.rand(1, self.n_videos, self.d_model))
        # self.rand_weights = torch.nn.Parameter(torch.ones((self.n_videos)), requires_grad=False )


    def forward(self, data_dict, return_cos_sim=False):
        x = data_dict["vis_feats"] # batch x set x dim
        bb, ss, dd = x.shape


        video_tag = self.video_tag[:, :ss, :]
        # video_tag_idxs = torch.multinomial(self.rand_weights, ss, replacement=False)
        # video_tag = torch.index_select(self.video_tag, -2, video_tag_idxs)


        x = x + video_tag[:ss]

        l = self.latent.repeat(bb, ss, 1)
        l = l + video_tag[:ss]

        encoder_input = torch.cat([l, x], dim=1)
        encoder_output = self.encoder(encoder_input)

        latent_output = encoder_output[:,:ss,:]
        latent_output = self.head_norm(latent_output)

        preds = self.p_head(latent_output)
        # preds = F.sigmoid(preds)
        data_dict["output"] = preds

        u_cls = self.cls_head(latent_output)
        u_cls = rearrange(u_cls, 'b s (p c) -> b s p c', p=self.n_prompts)
        data_dict["u_cls"] = u_cls

        return data_dict
    
class Psingle(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.d_model = 256
        self.d_feedforward = 512
        self.n_heads = 1
        self.n_layers = 1
        self.n_prompts = args.n_dataset_prompts
        self.n_videos = args.set_size
        # self.input_project = torch.nn.Linear(self.d_input, self.d_model)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self.d_model, dim_feedforward=self.d_feedforward, nhead=self.n_heads, batch_first=True)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)
        self.latent = torch.nn.Parameter(torch.zeros(1,1,self.d_model))
        self.head_norm = torch.nn.LayerNorm(self.d_model)
        self.cls_head = torch.nn.Linear(self.d_model, self.n_prompts)
        # self.video_tag = torch.nn.Parameter(torch.zeros(1, self.n_videos, self.d_model))

    def forward(self, data_dict, return_cos_sim=False):
        x = data_dict["vis_feats"] # batch x set x dim
        bb, ss, dd = x.shape
        # x = x + self.video_tag
        l = self.latent.repeat(bb, 1, 1)
        encoder_input = torch.cat([l, x], dim=1)
        encoder_output = self.encoder(encoder_input)
        latent_output = encoder_output[:,0,:]
        latent_output = self.head_norm(latent_output)
        preds = self.cls_head(latent_output)
        data_dict["output"] = preds

        return data_dict
    
class CSPred(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.d_model = args.d_model if hasattr(args, "d_model") else 256
        self.d_feedforward = args.d_ff if hasattr(args, "d_ff") else 512
        self.n_heads = 2
        self.n_layers = 4
        self.n_prompts = args.n_dataset_prompts
        self.n_videos = args.set_size

        # self.input_project = torch.nn.Linear(self.d_input, self.d_model)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self.d_model, dim_feedforward=self.d_feedforward, nhead=self.n_heads, batch_first=True)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)
        self.latent = torch.nn.Parameter(torch.zeros(1,1,self.d_model))
        self.head_norm = torch.nn.LayerNorm(self.d_model)
        self.cls_head = torch.nn.Linear(self.d_model, self.n_prompts)
        # self.video_tag = torch.nn.Parameter(torch.zeros(1, self.n_videos, self.d_model))

    def forward(self, data_dict, return_cos_sim=False):
        x = data_dict["vis_feats"] # batch x set x dim
        bb, ss, dd = x.shape
       # x = x + self.video_tag
        l = self.latent.repeat(bb, 1, 1)

        encoder_input = torch.cat([l, x], dim=1)
        encoder_output = self.encoder(encoder_input)
        latent_output = encoder_output[:,0,:]
        latent_output = self.head_norm(latent_output)
        output = self.cls_head(latent_output)

        print(output.shape)

        data_dict["output_cs_bpvv"] = output

        diag_amax = torch.arange(ss, device=output.device, requires_grad=False)
        diag_amax = rearrange(diag_amax, 's -> 1 1 s')
        diag_amax = diag_amax.repeat(bb, output.shape[-3], 1)

        p_amax_v = torch.argmax(output, dim=-1)
        p_amax_t = torch.argmax(output, dim=-2)
        p_is_unique_v = torch.eq(p_amax_v, diag_amax).long()#, torch.eq(amax_t, diag_amax)).long()
        p_is_unique_t = torch.eq(p_amax_t, diag_amax).long()#, torch.eq(amax_t, diag_amax)).long()
        p_is_unique = torch.where((p_is_unique_v + p_is_unique_t) == 2, 1.0, 0.0)

        data_dict["u_cls_bpv"] = p_is_unique

        return data_dict
    


class CE(torch.nn.Module):
    def __init__(self,args):
        super().__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.args = args
    def forward(self, data_dict):
        return_dict = {}
        output = data_dict["output"]
        target = data_dict["target"]

        bb, ss, cc = output.shape

        target_oh = F.one_hot(target, num_classes=cc).float()
        target_oh = target_oh.sum(dim=-2) / self.args.n_prompts

        output = rearrange(output, 'b s ...-> (b s) ...')
        target_oh = rearrange(target_oh, 'b s ...-> (b s) ...')

        loss = self.loss_fn(output, target_oh)
        return_dict["loss"] = loss

        preds = torch.topk(output, k=self.args.n_prompts, dim=-1)[1]
        return_dict["prompt_id_preds"] = rearrange(preds, '(v s) ...-> v s ...', s=ss)
        gt = rearrange(target, 'b s ...-> (b s) ...')

        correct = torch.tensor(0.0)
        p_counts = torch.zeros(cc)

        for i in range(preds.shape[0]):
            for j in range(preds.shape[1]):
                p_counts[preds[i,j]] += 1
                if preds[i,j] in gt[i]:
                    correct += 1

        acc = correct / (preds.shape[0] * preds.shape[1]) * 100
        p_counts = p_counts / p_counts.sum() * 100
        p_counts = p_counts.int()

        return_dict["acc"] = acc
        return_dict["p_fracs"] = p_counts

        return return_dict

class BCE(torch.nn.Module):
    def __init__(self,args):
        super().__init__()
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.args = args
    def forward(self, data_dict):
        return_dict = {}
        output = data_dict["output"]
        target = data_dict["target"]

        bb, ss, cc = output.shape

        target_oh = F.one_hot(target, num_classes=cc).float()
        target_oh = target_oh.sum(dim=-2)

        output = rearrange(output, 'b s ...-> (b s) ...')
        target_oh = rearrange(target_oh, 'b s ...-> (b s) ...')

        loss = self.loss_fn(output, target_oh)
        return_dict["loss"] = loss

        preds = torch.topk(output, k=self.args.n_prompts, dim=-1)[1]
        return_dict["prompt_id_preds"] = rearrange(preds, '(v s) ...-> v s ...', s=ss)
        gt = rearrange(target, 'b s ...-> (b s) ...')

        correct = torch.tensor(0.0)
        p_counts = torch.zeros(cc)

        for i in range(preds.shape[0]):
            for j in range(preds.shape[1]):
                p_counts[preds[i,j]] += 1
                if preds[i,j] in gt[i]:
                    correct += 1

        acc = correct / (preds.shape[0] * preds.shape[1]) * 100
        p_counts = p_counts / p_counts.sum() * 100
        p_counts = p_counts.int()

        return_dict["acc"] = acc
        return_dict["p_fracs"] = p_counts

        return return_dict

def merge_and_cos_sim(output, text_emb, vis_emb):
    bb, ss, cc = output.shape
    bb, ss, cc, dd = text_emb.shape

    #sigmoid on output

    # #multiply output with text_emb
    # comb = output.unsqueeze(-1) * text_emb
    # comb.transpose_(-2,-1)    
    # comb_text_emb = comb.sum(dim=-2)

    o = rearrange(output, 'b s c -> (b s) c')
    t = rearrange(text_emb, 'b s c d -> (b s) c d')
    c = o.unsqueeze(-1) * t
    comb_text_emb = c.sum(dim=-2)
    comb_text_emb = rearrange(comb_text_emb, '(b s) d -> b s d', s=ss)

    # serial_comb = torch.zeros([bb, ss, dd], device=output.device)
    # for b in range(bb):
    #     for s in range(ss):
    #         for c in range(cc):
    #             serial_comb[b,s] += output[b, s, c] * text_emb[b,s,c]

    # print(serial_comb.shape)
    # #check if comb_text_emb is similar to serial_comb
    # print(torch.allclose(comb_text_emb, serial_comb))
    # exit(0)

    v_t_sim = F.cosine_similarity(vis_emb.unsqueeze(-3), comb_text_emb.unsqueeze(-2), dim=-1).transpose(-1,-2)

    return v_t_sim

def clip_loss(v_t_sim, reduction='mean'):
    bb, vv, tt = v_t_sim.shape
    labels = torch.arange(vv, device=v_t_sim.device, requires_grad=False)
    # labels = labels.unsqueeze(0).repeat(bb*vv,1)
    labels = labels.repeat(bb)
    logits_per_image = rearrange(v_t_sim, 'b v t -> (b v) t')
    logits_per_text = rearrange(v_t_sim.transpose(-1, -2), 'b t v -> (b t) v')

    # print(labels)
    # print(logits_per_image.shape)
    # print(logits_per_text.shape)
    # print(labels.shape)

    logits_per_image_loss =  F.cross_entropy(logits_per_image, labels, reduction=reduction)  
    logits_per_text_loss = F.cross_entropy(logits_per_text, labels, reduction=reduction)
    clip_loss = (logits_per_image_loss + logits_per_text_loss) / 2
    return clip_loss


class SimPred(torch.nn.Module):
    def __init__(self,args=None):
        super().__init__()
        # self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.args = args
    def forward(self, data_dict):
        return_dict = {}
        output = data_dict["output"]
        text_emb = data_dict["target_text_feats"]
        vis_emb = data_dict["vis_feats"]

        bb, ss, cc, dd = text_emb.shape

        # v_t_sim2 = torch.zeros([bb, ss, ss], device=output.device)
        # for b in range(bb):
        #     for s in range(ss):
        #         for t in range(ss):
        #             v_t_sim2[b,s,t] = F.cosine_similarity(vis_emb[b,s].unsqueeze(0), comb_text_emb[b,t].unsqueeze(0))[0]
        # print(torch.allclose(v_t_sim, v_t_sim2))

        v_t_sim = merge_and_cos_sim(output, text_emb, vis_emb)



        cos_sim_loss = clip_loss(v_t_sim)
    


        # loss pulling towards 0 and 1
        # binary_loss = torch.pow(torch.pow(output, 2), 0.25) + torch.pow(torch.pow((1-output), 2), 0.5) - 0.5

        bl = output - 0.5
        bl = bl * math.sqrt(2)
        bl = torch.pow(bl, 4) - torch.pow(bl, 2) + 0.25

        binary_loss = bl.mean()

        # loss checking sum of logits is close to number of required prompts
        sum_loss = output.sum(dim=-1) - self.args.n_prompts
        sum_loss = sum_loss**2
        sum_loss = sum_loss.mean()

        return_dict["cos_sim_loss"] = cos_sim_loss
        return_dict["binary_loss"] = binary_loss
        return_dict["sum_loss"] = sum_loss

        loss = cos_sim_loss
        # return_dict["loss"] = loss
        # return_dict["loss"] = loss# + 0.1*binary_loss #sum_loss`` #0.1 * binary_loss #+ sum_loss
        return_dict["loss"] = 100.0 * binary_loss + cos_sim_loss + 10.0 * sum_loss
        with torch.no_grad():

            return_dict["pos_preds_avg"] = torch.mean(output)
            return_dict["pos_preds_large"] = torch.mean((output > 0.9).float())
            return_dict["pos_preds_small"] = torch.mean((output < 0.1).float())

            preds = torch.topk(output, k=self.args.n_prompts, dim=-1)[1]

            return_dict["prompt_id_preds"] = preds

            preds_oh = F.one_hot(preds, num_classes=cc).float().sum(dim=-2)
            v_t_sim = merge_and_cos_sim(preds_oh, text_emb, vis_emb)

            return_dict["pred_loss"] = clip_loss(v_t_sim)

            labels = torch.arange(ss, device=output.device)
            v_preds = torch.argmax(v_t_sim, dim=-1)
            t_preds = torch.argmax(v_t_sim, dim=-2)
            v_acc = torch.tensor(0.0, device=output.device)
            t_acc = torch.tensor(0.0, device=output.device)
            for b in range(bb):
                v_acc += (v_preds[b] == labels).sum() / ss
                t_acc += (t_preds[b] == labels).sum() / ss

            v_acc = v_acc / bb * 100
            t_acc = t_acc / bb * 100
            return_dict["v_acc"] = v_acc
            return_dict["t_acc"] = t_acc

        return return_dict
    
class SingleOnlineBestP(torch.nn.Module):
    def __init__(self,args=None):
        super().__init__()
        # self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.args = args
    def forward(self, data_dict):
        return_dict = {}
        output = data_dict["output"]
        text_emb = data_dict["target_text_feats"]
        vis_emb = data_dict["vis_feats"]

        bb, ss, cc, dd = text_emb.shape

        with torch.no_grad():
            p_cs = []
            for p in range(cc):
                cs = F.cosine_similarity(vis_emb.unsqueeze(-3), text_emb[:,:,p,:].unsqueeze(-2), dim=-1).transpose(-1,-2)
                p_cs.append(cs)
            p_cs = torch.stack(p_cs, dim= 1)



            p_cs_v = rearrange(p_cs, 'b p v t -> (b p v) t')
            p_cs_t = rearrange(p_cs, 'b p v t -> (b p t) v')
            clip_labels = torch.arange(ss, device=output.device, requires_grad=False)
            clip_labels = clip_labels.repeat(bb*cc)
            clip_loss = F.cross_entropy(p_cs_v, clip_labels, reduction='none') + F.cross_entropy(p_cs_t, clip_labels, reduction='none')
            clip_loss = clip_loss / 2
            clip_loss = rearrange(clip_loss, '(b p v) -> b p v', b=bb, p=cc, v=ss)
            clip_loss = torch.mean(clip_loss, dim=-1)
            gt_labels = torch.argmin(clip_loss, dim=-1)

        # return_dict["loss"] = F.cross_entropy(output, gt_labels)
        return_dict["loss"] = torch.mean(torch.pow(output - clip_loss, 2))


        with torch.no_grad():
            # preds = torch.argmax(output, dim=-1)
            preds = torch.argmin(output, dim=-1)

            return_dict["acc"] = torch.mean((preds == gt_labels).float()) * 100

            # print(preds, preds.shape)

            # preds = torch.topk(-output, k=1, dim=-1)[1]
            preds_dup = rearrange(preds, 'b -> b 1 1')
            preds_dup = preds_dup.repeat(1, ss, 1)
            # print(preds_dup, preds_dup.shape)
            return_dict["prompt_id_preds"] = preds_dup

        return return_dict

class OnlinePerVid(torch.nn.Module):
    def __init__(self,args=None):
        super().__init__()
        # self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.args = args
    def forward(self, data_dict):
        return_dict = {}
        output = data_dict["output"] # b v p
        text_emb = data_dict["target_text_feats"] # b v p d
        vis_emb = data_dict["vis_feats"] # b v d
        u_cls = data_dict["u_cls"] # b v p c where c=2 (i.e. not unique and unique logits)
        u_cls = rearrange(u_cls, 'b v p c -> b p v c')

        # print(output.shape, text_emb.shape, vis_emb.shape, u_cls.shape)


        # everything should be in order batch, prompt, video, text, dim, class

        bb, ss, cc, dd = text_emb.shape

        with torch.no_grad():
            p_cs = []
            for p in range(cc):
                cs = F.cosine_similarity(vis_emb.unsqueeze(-3), text_emb[:,:,p,:].unsqueeze(-2), dim=-1).transpose(-1,-2)
                p_cs.append(cs)
            p_cs = torch.stack(p_cs, dim= 1)

            amax_v = torch.argmax(p_cs, dim=-2)
            amax_t = torch.argmax(p_cs, dim=-1)

            diag_amax = torch.arange(ss, device=output.device, requires_grad=False)
            diag_amax = rearrange(diag_amax, 's -> 1 1 s')
            diag_amax = diag_amax.repeat(bb, cc, 1)
            is_unique_v = torch.eq(amax_v, diag_amax).long()#, torch.eq(amax_t, diag_amax)).long()
            is_unique_t = torch.eq(amax_t, diag_amax).long()#, torch.eq(amax_t, diag_amax)).long()
            is_unique = torch.where((is_unique_v + is_unique_t) == 2, 1, 0)
            is_unique_labs = rearrange(is_unique, 'b p v -> (b p v)')

            p_cs_v = rearrange(p_cs, 'b p v t -> (b p v) t')
            p_cs_t = rearrange(p_cs, 'b p v t -> (b p t) v')
            clip_labels = rearrange(diag_amax, 'b p v -> (b p v)')
            clip_loss = F.cross_entropy(p_cs_v, clip_labels, reduction='none') + F.cross_entropy(p_cs_t, clip_labels, reduction='none')
            clip_loss = clip_loss / 2
            clip_loss = rearrange(clip_loss, '(b p v) -> b p v', b=bb, p=cc, v=ss)  
            clip_labels = torch.argmin(clip_loss, dim=-2) 
            clip_labels_flat = rearrange(clip_labels, 'b v -> (b v)')         

            # print(p_cs.shape, clip_loss.shape)


            # p_cs_v = rearrange(p_cs, 'b p v t -> (b p v) t')
            # p_cs_t = rearrange(p_cs, 'b p v t -> (b p t) v')
            # clip_labels = torch.arange(ss, device=output.device, requires_grad=False)
            # clip_labels = clip_labels.repeat(bb*cc)
            # clip_loss = F.cross_entropy(p_cs_v, clip_labels, reduction='none') + F.cross_entropy(p_cs_t, clip_labels, reduction='none')
            # clip_loss = clip_loss / 2
            # clip_loss = rearrange(clip_loss, '(b p v) -> b p v', b=bb, p=cc, v=ss)
            # clip_loss = torch.mean(clip_loss, dim=-1)
            # gt_labels = torch.argmin(clip_loss, dim=-1)


        # print(is_unique.shape, u_cls.shape)

        u_cls_flat = rearrange(u_cls, 'b p v c -> (b p v) c')
        # u_cls_flat_probs = F.log_softmax(u_cls_flat, dim=-1)
        # is_u_loss = F.mse_loss(output_flat, is_unique_labs)
        is_u_loss = F.cross_entropy(u_cls_flat, is_unique_labs).cuda()
        # is_u_loss = F.binary_cross_entropy(u_cls_flat[:,1], is_unique_labs.float()).cuda()



        # return_dict["loss"] = F.cross_entropy(output, gt_labels)

        # output_flat = rearrange(output, 'b v p -> (b v p)')
        # output_sm_bvp = F.softmax(output, dim=-1)
        # print(output_sm_bvp.shape, clip_loss.shape)
        # weighted_sum_output = (output_sm_bvp * rearrange(clip_loss, 'b p v -> b v p')).sum(dim=-1)
        # weighted_sum_loss = torch.mean(weighted_sum_output)

        # print(output.shape, clip_labels_flat.shape)

        
        p_classification_loss = F.cross_entropy(rearrange(output, 'b v p -> (b v) p'), clip_labels_flat)

        # weighted_sum_loss


        return_dict["u_loss"] = is_u_loss
        return_dict["p_loss"] = p_classification_loss
        return_dict["loss"] = is_u_loss #+ p_classification_loss


        with torch.no_grad():
            preds_u = torch.argmax(u_cls, dim=-1)
            # preds_u_probs = u_cls[:,:,:,1]
            preds_u_probs = F.softmax(u_cls, dim=-1)[:,:,:,1]
            # print(preds_u.shape, is_unique.shape)
            correct_u = torch.where(preds_u == is_unique, 1, 0)

            # ground truth - how many videos do have a unique prompt?
            return_dict["gt_unique"] = is_unique_labs.float().mean()

            # accuracy of binary unique classification
            preds_u_flat = torch.argmax(u_cls_flat, dim=-1)
            return_dict["u_acc"] = torch.mean((preds_u_flat == is_unique_labs).float()) * 100
            return_dict["pred_unique"] = torch.where(preds_u_flat == 1, 1, 0).sum() / u_cls_flat.shape[0] * 100
            return_dict["pred_not_unique"] = torch.where(preds_u_flat == 0, 1, 0).sum() / u_cls_flat.shape[0] * 100

            # accuracy of best prompt prediction per video
            preds_p = torch.argmax(output, dim=-1)
            labs_p = torch.argmin(clip_loss, dim=-2)
            return_dict["p_acc"] = torch.mean((preds_p == labs_p).float()) * 100

            # accuracy of uniqueness, when chosen by maximum prompt prediction probability
            chosen_u_acc = rearrange(correct_u, 'b p  v-> (b v) p').gather(1, rearrange(preds_p, 'b v -> (b v) 1')).float().mean() * 100
            return_dict["u_acc_chosen"] = chosen_u_acc

            # accuracy of uniqueness, when chosen by maximum uniqueness confidence
            max_probs = torch.argmax(preds_u_probs, dim=-2)
            max_prob_correct = rearrange(correct_u, 'b p  v-> (b v) p').gather(1, rearrange(max_probs, 'b v -> (b v) 1'))
            max_prob_acc = max_prob_correct.float().mean() * 100
            return_dict["u_acc_max_prob"] = max_prob_acc

            # accuracy of uniquencess per prompt, but only if above a threshold
            confidence_threshold = 0.9
            high_confidence = torch.where(preds_u_probs > confidence_threshold, 1, 0)
            n_high_confidence = torch.tensor(max(high_confidence.sum(), 1), device=output.device, requires_grad=False)
            high_confidence_acc = torch.logical_and(correct_u, high_confidence).float().sum()/n_high_confidence * 100
            return_dict["u_acc_conf"] = high_confidence_acc
            return_dict["n_u_acc_conf"] = n_high_confidence

            #accuracy of max unique prompt per video, above a threshold. Also the number of videos which are designated unique.
            # print(high_confidence.shape, correct_u.shape, preds_u_probs.shape, preds_p.shape)

            max_high_confidence = rearrange(high_confidence, 'b p  v-> (b v) p').gather(1, rearrange(max_probs, 'b v -> (b v) 1'))
            n_max_high_confidence = torch.tensor(max(max_high_confidence.sum(), 1), device=output.device, requires_grad=False)
            n_vids = torch.tensor(max_high_confidence.shape[0], device=output.device, requires_grad=False)


            max_high_confidence_acc = torch.logical_and(max_high_confidence, max_prob_correct).float().sum()/n_max_high_confidence * 100

            return_dict["u_acc_max_conf"] = max_high_confidence_acc
            return_dict["n_u_acc_max_conf"] = n_max_high_confidence
            return_dict["n_vids"] = n_vids
            # chosen_u_probs = rearrange(preds_u_probs, 'b p  v-> (b v) p').gather(1, rearrange(preds_p, 'b v -> (b v) 1')).float()

            # print(high_confidence_acc)


            # accuracy of max uniqueness per video, above a threshold.


            # # preds = torch.topk(-output, k=1, dim=-1)[1]
            # preds_dup = rearrange(preds, 'b -> b 1 1')
            # preds_dup = preds_dup.repeat(1, ss, 1)
            # # print(preds_dup, preds_dup.shape)
            return_dict["prompt_id_preds"] = torch.zeros(bb, ss)
            return_dict["prompt_id_gt"] = rearrange(is_unique, 'b p v -> b v p')

        return return_dict

class CSPairClsP(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.d_model = args.d_model if hasattr(args, "d_model") else 256
        self.d_feedforward = args.d_ff if hasattr(args, "d_ff") else 512
        self.pp_feat_str = args.pp_feat_str if hasattr(args, "pp_feat_str") else "vis_feats"
        self.n_heads = args.pp_nh if hasattr(args, "pp_nh") else 4
        self.n_layers = args.pp_nl if hasattr(args, "pp_nl") else 2
        self.n_prompts = args.n_dataset_prompts
        self.n_videos = args.set_size

        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self.d_model, dim_feedforward=self.d_feedforward, nhead=self.n_heads, batch_first=True)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)
        self.latent = torch.nn.Parameter(torch.zeros(1,self.n_prompts,self.d_model))

        self.head_norm = torch.nn.LayerNorm(self.d_model)
        self.cls_head = torch.nn.Linear(self.d_model, self.n_prompts*2)
        self.p_head = torch.nn.Linear(self.d_model, 1)

        self.video_tag = torch.nn.Parameter(torch.zeros(1, self.n_videos, self.d_model))

    def forward(self, data_dict, return_cos_sim=False, pp_max_batch=9999999):
        x = data_dict[self.pp_feat_str] # batch x set x dim
        bb, ss, dd = x.shape

        v1 = torch.repeat_interleave(x, repeats=ss, dim=-2).unsqueeze(-2)
        v2 = x.repeat(1, ss, 1).unsqueeze(-2)
        x = torch.cat([v1, v2], dim=-2)

        x = rearrange(x, 'b ss ... -> (b ss) ...')

        video_tag = self.video_tag[:, :2, :]
        x = x + video_tag[:ss]

        # l = self.latent.repeat(bb * ss * ss, 1, 1)
        # print(hasattr(self.args, "pp_max_batch"))
        # print(x.shape[0])
        # print(self.args.pp_/max_batch)

        if x.shape[0] > pp_max_batch:
            # encoder_output = []
            xsplit = torch.split(x, pp_max_batch, dim=0)
            output = []
            for xx in xsplit:
                ll = self.latent.expand(xx.shape[0], -1, -1)
                encoder_input = torch.cat([ll, xx], dim=1)
                encoder_output = self.encoder(encoder_input)
                latent_output = encoder_output[:,:self.n_prompts,:]
                latent_output = self.head_norm(latent_output)
                output.append(self.p_head(latent_output))
            output = torch.cat(output, dim=0)

        else:
            l = self.latent.expand(bb * ss * ss, -1, -1)
            encoder_input = torch.cat([l, x], dim=1)
            encoder_output = self.encoder(encoder_input)

            latent_output = encoder_output[:,:self.n_prompts,:]

            latent_output = self.head_norm(latent_output)
            output = self.p_head(latent_output)

        output = rearrange(output, '(b v1 v2) p 1 -> b p v1 v2', b=bb, v1=ss, v2=ss)
        data_dict["output_cs_bpvv"] = output

        diag_amax = torch.arange(ss, device=output.device, requires_grad=False)
        diag_amax = rearrange(diag_amax, 's -> 1 1 s')
        diag_amax = diag_amax.repeat(bb, output.shape[-3], 1)

        p_amax_v = torch.argmax(output, dim=-1)
        p_amax_t = torch.argmax(output, dim=-2)
        p_is_unique_v = torch.eq(p_amax_v, diag_amax).long()#, torch.eq(amax_t, diag_amax)).long()
        p_is_unique_t = torch.eq(p_amax_t, diag_amax).long()#, torch.eq(amax_t, diag_amax)).long()
        p_is_unique = torch.where((p_is_unique_v + p_is_unique_t) == 2, 1.0, 0.0)

        data_dict["u_cls_bpv"] = p_is_unique

        return data_dict


class CSPairMLP(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.d_model = args.d_model if hasattr(args, "d_model") else 256
        self.d_feedforward = args.d_ff if hasattr(args, "d_ff") else 512
        self.pp_feat_str = args.pp_feat_str if hasattr(args, "pp_feat_str") else "vis_feats"
        self.n_heads = args.pp_nh if hasattr(args, "pp_nh") else 4
        self.n_layers = args.pp_nl if hasattr(args, "pp_nl") else 2
        self.n_prompts = args.n_dataset_prompts
        self.n_videos = args.set_size

        self.linear1 = torch.nn.Linear(self.d_model * 2, self.d_feedforward)
        self.linear2 = torch.nn.Linear(self.d_feedforward, int(self.d_feedforward / 4))
        self.linear3 = torch.nn.Linear(int(self.d_feedforward / 4), self.n_prompts)
        self.ln1 = torch.nn.LayerNorm(self.d_model * 2)
        self.ln2 = torch.nn.LayerNorm(self.d_feedforward)
        self.ln3 = torch.nn.LayerNorm(int(self.d_feedforward / 4))
        self.gelu1 = torch.nn.GELU()
        self.gelu2 = torch.nn.GELU()


    def forward(self, data_dict, return_cos_sim=False):
        x = data_dict[self.pp_feat_str] # batch x set x dim
        bb, ss, dd = x.shape

        v1 = torch.repeat_interleave(x, repeats=ss, dim=-2).unsqueeze(-2)
        v2 = x.repeat(1, ss, 1).unsqueeze(-2)
        x = torch.cat([v1, v2], dim=-2)

        x = rearrange(x, 'b ss n d -> (b ss) (n d)')

        x = self.gelu1(self.linear1(self.ln1(x)))
        x = self.gelu2(self.linear2(self.ln2(x)))
        x = self.linear3(self.ln3(x))
        output = x

        output = rearrange(output, '(b v1 v2) p -> b p v1 v2', b=bb, v1=ss, v2=ss)
        data_dict["output_cs_bpvv"] = output

        diag_amax = torch.arange(ss, device=output.device, requires_grad=False)
        diag_amax = rearrange(diag_amax, 's -> 1 1 s')
        diag_amax = diag_amax.repeat(bb, output.shape[-3], 1)

        p_amax_v = torch.argmax(output, dim=-1)
        p_amax_t = torch.argmax(output, dim=-2)
        p_is_unique_v = torch.eq(p_amax_v, diag_amax).long()#, torch.eq(amax_t, diag_amax)).long()
        p_is_unique_t = torch.eq(p_amax_t, diag_amax).long()#, torch.eq(amax_t, diag_amax)).long()
        p_is_unique = torch.where((p_is_unique_v + p_is_unique_t) == 2, 1.0, 0.0)

        data_dict["u_cls_bpv"] = p_is_unique

        return data_dict


class CSPair(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.d_model = args.d_model if hasattr(args, "d_model") else 256
        self.d_feedforward = args.d_ff if hasattr(args, "d_ff") else 512
        self.pp_feat_str = args.pp_feat_str if hasattr(args, "pp_feat_str") else "vis_feats"
        self.n_heads = args.pp_nh if hasattr(args, "pp_nh") else 4
        self.n_layers = args.pp_nl if hasattr(args, "pp_nl") else 2
        self.n_prompts = args.n_dataset_prompts
        self.n_videos = args.set_size

        # self.input_project = torch.nn.Linear(self.d_input, self.d_model)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self.d_model, dim_feedforward=self.d_feedforward, nhead=self.n_heads, batch_first=True)
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)
        self.latent = torch.nn.Parameter(torch.zeros(1,1,self.d_model))

        self.head_norm = torch.nn.LayerNorm(self.d_model)
        self.cls_head = torch.nn.Linear(self.d_model, self.n_prompts*2)
        self.p_head = torch.nn.Linear(self.d_model, self.n_prompts)

        self.video_tag = torch.nn.Parameter(torch.zeros(1, self.n_videos, self.d_model))
        # self.video_tag = torch.nn.Parameter(torch.rand(1, self.n_videos, self.d_model))
        # self.rand_weights = torch.nn.Parameter(torch.ones((self.n_videos)), requires_grad=False )


    def forward(self, data_dict, return_cos_sim=False):
        x = data_dict[self.pp_feat_str] # batch x set x dim
        bb, ss, dd = x.shape

        # for each video1/video2 pair, for each prompt, predict cos sim between video1 and caption of video1, and video1 and caption of video2
        # pass in CLS, and tagged video 1 and tagged video 2.
        # take linear layer from CLS to n_prompts output

        v1 = torch.repeat_interleave(x, repeats=ss, dim=-2).unsqueeze(-2)
        v2 = x.repeat(1, ss, 1).unsqueeze(-2)
        x = torch.cat([v1, v2], dim=-2)

        x = rearrange(x, 'b ss ... -> (b ss) ...')

        video_tag = self.video_tag[:, :2, :]
        # video_tag_idxs = torch.multinomial(self.rand_weights, ss, replacement=False)
        # video_tag = torch.index_select(self.video_tag, -2, video_tag_idxs)
        x = x + video_tag[:ss]

        l = self.latent.repeat(bb * ss * ss, 1, 1)

        encoder_input = torch.cat([l, x], dim=1)
        encoder_output = self.encoder(encoder_input)

        latent_output = encoder_output[:,0,:]
        latent_output = self.head_norm(latent_output)
        output = self.p_head(latent_output)


        output = rearrange(output, '(b v1 v2) p -> b p v1 v2', b=bb, v1=ss, v2=ss)

        # print(output.shape)

        data_dict["output_cs_bpvv"] = output


        diag_amax = torch.arange(ss, device=output.device, requires_grad=False)
        diag_amax = rearrange(diag_amax, 's -> 1 1 s')
        diag_amax = diag_amax.repeat(bb, output.shape[-3], 1)

        p_amax_v = torch.argmax(output, dim=-1)
        p_amax_t = torch.argmax(output, dim=-2)
        p_is_unique_v = torch.eq(p_amax_v, diag_amax).long()#, torch.eq(amax_t, diag_amax)).long()
        p_is_unique_t = torch.eq(p_amax_t, diag_amax).long()#, torch.eq(amax_t, diag_amax)).long()
        p_is_unique = torch.where((p_is_unique_v + p_is_unique_t) == 2, 1.0, 0.0)

        data_dict["u_cls_bpv"] = p_is_unique

        return data_dict






class CSPairLoss(torch.nn.Module):
    def __init__(self,args=None):
        super().__init__()
        # self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.args = args
    def forward(self, data_dict):
        return_dict = {}
        output = data_dict["output_cs_bpvv"] # b p v1 v2
        text_emb = data_dict["target_text_feats"] # b v p d
        vis_emb = data_dict["vis_feats"] # b v d
        u_cls = data_dict["u_cls_bpv"] # b p v


        bb, ss, cc, dd = text_emb.shape

        with torch.no_grad():
            gt_cs = []
            for p in range(cc):
                cs = F.cosine_similarity(vis_emb.unsqueeze(-3), text_emb[:,:,p,:].unsqueeze(-2), dim=-1).transpose(-1,-2)
                gt_cs.append(cs)
            gt_cs = torch.stack(gt_cs, dim= 1)


            amax_v = torch.argmax(gt_cs, dim=-1)
            amax_t = torch.argmax(gt_cs, dim=-2)

            diag_amax = torch.arange(ss, device=output.device, requires_grad=False)
            diag_amax = rearrange(diag_amax, 's -> 1 1 s')
            diag_amax = diag_amax.repeat(bb, cc, 1)
            
            is_unique_v = torch.eq(amax_v, diag_amax).long()#, torch.eq(amax_t, diag_amax)).long()
            is_unique_t = torch.eq(amax_t, diag_amax).long()#, torch.eq(amax_t, diag_amax)).long()
            is_unique = torch.where((is_unique_v + is_unique_t) == 2, 1, 0)



            is_unique_labs = rearrange(is_unique, 'b p v -> (b p v)')

            gt_cs_v = rearrange(gt_cs, 'b p v t -> (b p v) t')
            gt_cs_t = rearrange(gt_cs, 'b p v t -> (b p t) v')
            clip_labels = rearrange(diag_amax, 'b p v -> (b p v)')
            clip_loss = F.cross_entropy(gt_cs_v, clip_labels, reduction='none') + F.cross_entropy(gt_cs_t, clip_labels, reduction='none')
            clip_loss = clip_loss / 2
            clip_loss = rearrange(clip_loss, '(b p v) -> b p v', b=bb, p=cc, v=ss)  
            clip_labels = torch.argmin(clip_loss, dim=-2) 
            clip_labels_flat = rearrange(clip_labels, 'b v -> (b v)')         



        diff = output - gt_cs
        diff = diff**2
        avg_diff = diff.mean()
        return_dict["loss"] = avg_diff


        # print(is_unique.shape, u_cls.shape)

        # u_cls_flat = rearrange(u_cls, 'b p v c -> (b p v) c')
        # # u_cls_flat_probs = F.log_softmax(u_cls_flat, dim=-1)
        # # is_u_loss = F.mse_loss(output_flat, is_unique_labs)
        # is_u_loss = F.cross_entropy(u_cls_flat, is_unique_labs).cuda()
        # # is_u_loss = F.binary_cross_entropy(u_cls_flat[:,1], is_unique_labs.float()).cuda()



        with torch.no_grad():

            # p_amax_v = torch.argmax(output, dim=-1)
            # p_amax_t = torch.argmax(output, dim=-2)
            # p_is_unique_v = torch.eq(p_amax_v, diag_amax).long()#, torch.eq(amax_t, diag_amax)).long()
            # p_is_unique_t = torch.eq(p_amax_t, diag_amax).long()#, torch.eq(amax_t, diag_amax)).long()
            # p_is_unique = torch.where((p_is_unique_v + p_is_unique_t) == 2, 1, 0)
            # preds_u = p_is_unique

            preds_u = u_cls

            correct_u = torch.where(preds_u == is_unique, 1.0, 0.0)

            return_dict["prompt_id_preds"] = torch.zeros(bb, ss)
            return_dict["prompt_id_gt"] = rearrange(is_unique, 'b p v -> b v p')

            # ground truth - how many videos do have a unique prompt?
            return_dict["gt_unique"] = is_unique_labs.float().mean() * 100

            # accuracy of binary unique classification
            return_dict["u_acc"] = torch.mean(correct_u.float()) * 100
            return_dict["pred_unique"] = torch.where(preds_u == 1, 1.0, 0.0).sum() / (bb*ss*cc) * 100
            return_dict["pred_not_unique"] = torch.where(preds_u == 0, 1.0, 0.0).sum() / (bb*ss*cc) * 100

            return_dict["diff"] = torch.cat([output, gt_cs], dim=-1)

            pred_hist = torch.zeros(10)
            gt_hist = torch.zeros(10)
            for i in range(10):
                pred_hist[i] = torch.where(output < 0.1*(i+1), 1.0, 0.0).mean()
                gt_hist[i] = torch.where(gt_cs < 0.1*(i+1), 1.0, 0.0).mean()
            return_dict["pred_hist"] = pred_hist
            return_dict["gt_hist"] = gt_hist






            return return_dict

            # accuracy of best prompt prediction per video
            preds_p = torch.argmax(output, dim=-1)
            labs_p = torch.argmin(clip_loss, dim=-2)
            return_dict["p_acc"] = torch.mean((preds_p == labs_p).float()) * 100

            # accuracy of uniqueness, when chosen by maximum prompt prediction probability
            chosen_u_acc = rearrange(correct_u, 'b p  v-> (b v) p').gather(1, rearrange(preds_p, 'b v -> (b v) 1')).float().mean() * 100
            return_dict["u_acc_chosen"] = chosen_u_acc

            # accuracy of uniqueness, when chosen by maximum uniqueness confidence
            max_probs = torch.argmax(preds_u_probs, dim=-2)
            max_prob_correct = rearrange(correct_u, 'b p  v-> (b v) p').gather(1, rearrange(max_probs, 'b v -> (b v) 1'))
            max_prob_acc = max_prob_correct.float().mean() * 100
            return_dict["u_acc_max_prob"] = max_prob_acc

            # accuracy of uniquencess per prompt, but only if above a threshold
            confidence_threshold = 0.9
            high_confidence = torch.where(preds_u_probs > confidence_threshold, 1, 0)
            n_high_confidence = torch.tensor(max(high_confidence.sum(), 1), device=output.device, requires_grad=False)
            high_confidence_acc = torch.logical_and(correct_u, high_confidence).float().sum()/n_high_confidence * 100
            return_dict["u_acc_conf"] = high_confidence_acc
            return_dict["n_u_acc_conf"] = n_high_confidence

            #accuracy of max unique prompt per video, above a threshold. Also the number of videos which are designated unique.
            # print(high_confidence.shape, correct_u.shape, preds_u_probs.shape, preds_p.shape)

            max_high_confidence = rearrange(high_confidence, 'b p  v-> (b v) p').gather(1, rearrange(max_probs, 'b v -> (b v) 1'))
            n_max_high_confidence = torch.tensor(max(max_high_confidence.sum(), 1), device=output.device, requires_grad=False)
            n_vids = torch.tensor(max_high_confidence.shape[0], device=output.device, requires_grad=False)


            max_high_confidence_acc = torch.logical_and(max_high_confidence, max_prob_correct).float().sum()/n_max_high_confidence * 100

            return_dict["u_acc_max_conf"] = max_high_confidence_acc
            return_dict["n_u_acc_max_conf"] = n_max_high_confidence
            return_dict["n_vids"] = n_vids
            # chosen_u_probs = rearrange(preds_u_probs, 'b p  v-> (b v) p').gather(1, rearrange(preds_p, 'b v -> (b v) 1')).float()

            # print(high_confidence_acc)


            # accuracy of max uniqueness per video, above a threshold.


            # # preds = torch.topk(-output, k=1, dim=-1)[1]
            # preds_dup = rearrange(preds, 'b -> b 1 1')
            # preds_dup = preds_dup.repeat(1, ss, 1)
            # # print(preds_dup, preds_dup.shape)


        return return_dict



    
if __name__ == '__main__':
    class Args:
        def __init__(self):
            self.n_prompts = 2
            self.n_dataset_prompts = 5
            self.set_size = 3
            self.batch_size = 2
    args = Args()

    d = {}

    # d["vis_feats"] = torch.rand(2, 3, 4)
    # d["output"] = torch.rand(2, 3, 5)
    # d["u_cls"] = torch.rand(2, 3, 5, 2)
    # d["target_text_feats"] = torch.rand(2, 3, 5, 4)
    d["vis_feats"] = torch.rand(2, 3, 256).cuda()
    # d["output"] = torch.rand(2, 3, 5).cuda()
    d["target_text_feats"] = torch.rand(2, 3, 5, 256).cuda()

    # model = P1(args).cuda()
    # loss_fn = OnlinePerVid(args).cuda()


    # d["output"] = torch.rand(2, 5, 3, 3)
    model = CSPairMLP(args).cuda()
    # model = CSPairClsP(args).cuda()
    # model = CSPair(args).cuda()
    # model = CSPred(args).cuda()
    loss_fn = CSPairLoss(args).cuda()












    # model = P1(args)
    # model.cuda()
    # loss_fn = OnlinePerVid(args)



    d = model(d)


    # d = loss_fn(d)

    for k, v in d.items():
        if len(v.shape) > 1:
            print(k, v.shape)
        else:
            print(k, v)