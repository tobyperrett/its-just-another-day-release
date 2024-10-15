import torch
from einops import rearrange
import torch.nn.functional as F

class MLP(torch.nn.Module):
    def __init__(self,):
        super().__init__()
        self.ln1 = torch.nn.LayerNorm(768 * 2)
        self.linear1 = torch.nn.Linear(768 * 2, 768)
        self.gelu1 = torch.nn.GELU()
        self.ln2 = torch.nn.LayerNorm(768)
        self.linear2 = torch.nn.Linear(768, 1)
        torch.nn.init.normal_(self.linear1.weight.data, std=0.01)
        torch.nn.init.normal_(self.linear2.weight.data, std=0.01)
    
    def forward(self, x):
        x = self.gelu1(self.linear1(self.ln1(x)))
        x = self.linear2(self.ln2(x))
        return x
    
class MLP_2LN(torch.nn.Module):
    def __init__(self,):
        super().__init__()
        self.ln11 = torch.nn.LayerNorm(768)
        self.ln12 = torch.nn.LayerNorm(768)
        self.linear1 = torch.nn.Linear(768 * 2, 768)
        self.gelu1 = torch.nn.GELU()
        self.ln2 = torch.nn.LayerNorm(768)
        self.linear2 = torch.nn.Linear(768, 1)
        torch.nn.init.normal_(self.linear1.weight.data, std=0.01)
        torch.nn.init.normal_(self.linear2.weight.data, std=0.01)
    
    def forward(self, data_dict):
        x1 = data_dict["vis_feats"]
        x1 = data_dict["bos_feats"]

        x1 = self.ln11(x1)
        x2 = self.ln12(x2)
        x = torch.concat([x1,x2], dim=-1)
        x = self.gelu1(self.linear1(x))
        x = self.linear2(self.ln2(x))
        return x
    
class TP_MLP1(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.ln11 = torch.nn.LayerNorm(768)
        self.ln12 = torch.nn.LayerNorm(768)
        self.linear1 = torch.nn.Linear(768 * 2, 768)
        self.gelu1 = torch.nn.GELU()
        self.ln2 = torch.nn.LayerNorm(768)
        self.linear2 = torch.nn.Linear(768, 1)
        torch.nn.init.normal_(self.linear1.weight.data, std=0.01)
        torch.nn.init.normal_(self.linear2.weight.data, std=0.01)
    
    def forward(self, data_dict):

        x1 = data_dict["all_vis_feats"]
        x2 = data_dict["all_bos_feats"]
        x1 = torch.mean(x1, dim=-2)

        x1 = self.ln11(x1)
        x2 = self.ln12(x2)
        x = torch.concat([x1,x2], dim=-1)
        x = self.gelu1(self.linear1(x))
        x = self.linear2(self.ln2(x))

        data_dict["output"] = x.squeeze(-1)
        return data_dict

class TP_MLP2(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # vision estimate
        self.ln11 = torch.nn.LayerNorm(768 * self.args.n_img_q)
        self.linear1 = torch.nn.Linear(768 * self.args.n_img_q, 768)
        self.gelu1 = torch.nn.GELU()
        self.ln2 = torch.nn.LayerNorm(768)
        self.linear2 = torch.nn.Linear(768, 256)
        torch.nn.init.normal_(self.linear1.weight.data, std=0.01)
        torch.nn.init.normal_(self.linear2.weight.data, std=0.01)

        # text estimate
        self.t_ln11 = torch.nn.LayerNorm(768)
        self.t_linear1 = torch.nn.Linear(768, 768)
        self.t_gelu1 = torch.nn.GELU()
        self.t_ln2 = torch.nn.LayerNorm(768)
        self.t_linear2 = torch.nn.Linear(768, 256)
    
    def forward(self, data_dict, return_cos_sim=False):
        #vision estimate
        x = data_dict["vis_feats"]
        x = rearrange(x, 'b t d -> b (t d)')
        x = self.ln11(x)
        x = self.gelu1(self.linear1(x))
        x = self.linear2(self.ln2(x))
        data_dict["vis_pred"] = x

        #text estimate
        t = data_dict["bos_feats"]
        t = self.t_ln11(t)
        t = self.t_gelu1(self.t_linear1(t))
        t = self.t_linear2(self.t_ln2(t))
        data_dict["text_pred"] = t

        if return_cos_sim:
            c = torch.cosine_similarity(x, t, dim=-1)
            data_dict["cos_sim"] = c

        return data_dict
    
class TP_MLP3(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        self.ln11 = torch.nn.LayerNorm(768 * self.args.n_img_q)
        self.ln12 = torch.nn.LayerNorm(768)
        self.linear1 = torch.nn.Linear(768 * (self.args.n_img_q + 1), 768)
        self.gelu1 = torch.nn.GELU()
        self.ln2 = torch.nn.LayerNorm(768)
        self.linear2 = torch.nn.Linear(768, 1)
        torch.nn.init.normal_(self.linear1.weight.data, std=0.01)
        torch.nn.init.normal_(self.linear2.weight.data, std=0.01)
    
    def forward(self, data_dict):

        x1 = data_dict["all_vis_feats"]
        x2 = data_dict["all_bos_feats"]
        x1 = rearrange(x1, 'b t d -> b (t d)')

        x1 = self.ln11(x1)
        x2 = self.ln12(x2)
        x = torch.concat([x1,x2], dim=-1)
        x = self.gelu1(self.linear1(x))
        x = self.linear2(self.ln2(x))

        data_dict["output"] = x.squeeze(-1)
        return data_dict

class TP_T1(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.combined=False
        self.args = args

        # vision estimate
        v_encoder_layer = torch.nn.TransformerEncoderLayer(d_model=768, nhead=4, batch_first=True)
        self.v_encoder = torch.nn.TransformerEncoder(v_encoder_layer, num_layers=2)
        self.v_latent = torch.nn.Parameter(torch.zeros(1,1,768))
        self.v_head = torch.nn.Linear(768, 256)

        # self.t_pe = torch.nn.Parameter(torch.zeros(1, self.args.n_img_q + 1, 768))
        self.t_pe = torch.nn.Parameter(torch.zeros(1, self.args.n_img_q + 2, 768))
        t_encoder_layer = torch.nn.TransformerEncoderLayer(d_model=768, nhead=4, batch_first=True)
        self.t_encoder = torch.nn.TransformerEncoder(t_encoder_layer, num_layers=2)
        self.t_latent = torch.nn.Parameter(torch.zeros(1,1,768))
        self.t_head = torch.nn.Linear(768, 256)

        # # text estimate
        # self.t_ln11 = torch.nn.LayerNorm(768)
        # self.t_linear1 = torch.nn.Linear(768, 768)
        # self.t_gelu1 = torch.nn.GELU()
        # self.t_ln2 = torch.nn.LayerNorm(768)
        # self.t_linear2 = torch.nn.Linear(768, 768)
        # self.t_gelu2 = torch.nn.GELU()
        # self.t_ln3 = torch.nn.LayerNorm(768)
        # self.t_linear3 = torch.nn.Linear(768, 256)


    def forward(self, data_dict, return_cos_sim=False):
        #vision estimate
        x = data_dict["vis_feats"]
        l = self.v_latent.repeat(x.shape[0], 1, 1)
        x = torch.cat([l, x], dim=1)
        x = self.v_encoder(x)
        x = x[:,0,:]
        x = self.v_head(x)
        data_dict["vis_pred"] = x

        t = data_dict["bos_feats"]
        t = rearrange(t, 'b d -> b 1 d')
        lt = self.t_latent.repeat(t.shape[0], 1, 1)


        t = torch.cat([lt, t], dim=1)
        t = torch.cat([t, data_dict["vis_feats"]], dim=1)
        t = t + self.t_pe[:, :t.shape[1], :]
        t = self.t_encoder(t)
        t = t[:,0,:]
        t = self.t_head(t)
        data_dict["text_pred"] = t


        # #text estimate
        # t = data_dict["bos_feats"]
        # t = self.t_ln11(t)
        # t = self.t_gelu1(self.t_linear1(t))
        # t = self.t_gelu2(self.t_linear2(self.t_ln2(t)))
        # t = self.t_linear3(self.t_ln3(t))
        # data_dict["text_pred"] = t

        if return_cos_sim:
            c = torch.cosine_similarity(x, t, dim=-1)
            data_dict["cos_sim"] = c

        return data_dict
    
class TP_T2(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.combined=True
        
        self.args = args
        
        self.t_pe = torch.nn.Parameter(torch.zeros(1, self.args.n_img_q + 2, 768))
        t_encoder_layer = torch.nn.TransformerEncoderLayer(d_model=768, nhead=4, batch_first=True)
        self.t_encoder = torch.nn.TransformerEncoder(t_encoder_layer, num_layers=2)
        self.t_latent = torch.nn.Parameter(torch.zeros(1,1,768))
        self.t_head1 = torch.nn.Linear(768, 256)
        self.t_head_gelu = torch.nn.GELU()
        self.t_head2 = torch.nn.Linear(256, 1)

    def forward(self, data_dict, return_cos_sim=False):

        x = data_dict["all_vis_feats"]
        t = data_dict["all_bos_feats"]
        t = rearrange(t, 'b d -> b 1 d')
        lt = self.t_latent.repeat(t.shape[0], 1, 1)
        t = torch.cat([lt, t, x], dim=1)
        t = t + self.t_pe[:, :t.shape[1], :]
        t = self.t_encoder(t)
        t = t[:,0,:]
        t = self.t_head1(t)
        t = self.t_head_gelu(t)
        t = self.t_head2(t)
        data_dict["output"] = t.squeeze(-1)

        return data_dict
    
class TP_T3(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.combined=False


        self.args = args

        # vision estimate
        v_encoder_layer = torch.nn.TransformerEncoderLayer(d_model=768, nhead=4, batch_first=True)
        self.v_encoder = torch.nn.TransformerEncoder(v_encoder_layer, num_layers=2)
        self.v_latent = torch.nn.Parameter(torch.zeros(1,1,768))
        self.v_head = torch.nn.Linear(768, 256)

        # self.t_pe = torch.nn.Parameter(torch.zeros(1, self.args.n_img_q + 1, 768))
        self.t_pe = torch.nn.Parameter(torch.zeros(1, self.args.n_img_q + 2, 768))
        t_encoder_layer = torch.nn.TransformerEncoderLayer(d_model=768, nhead=4, batch_first=True)
        self.t_encoder = torch.nn.TransformerEncoder(t_encoder_layer, num_layers=2)
        self.t_latent = torch.nn.Parameter(torch.zeros(1,1,768))
        self.t_head = torch.nn.Linear(768, 256)

    def forward(self, data_dict, return_cos_sim=False):
        #vision estimate
        x = data_dict["vis_feats"]
        l = self.v_latent.repeat(x.shape[0], 1, 1)
        x = torch.cat([l, x], dim=1)
        x = self.v_encoder(x)
        x = x[:,0,:]
        x = self.v_head(x)
        data_dict["vis_pred"] = x

        t = data_dict["bos_feats"]
        t = rearrange(t, 'b d -> b 1 d')
        lt = self.t_latent.repeat(t.shape[0], 1, 1)

        t = torch.cat([lt, t], dim=1)
        # t = torch.cat([t, data_dict["vis_feats"]], dim=1)
        t = t + self.t_pe[:, :t.shape[1], :]
        t = self.t_encoder(t)
        t = t[:,0,:]
        t = self.t_head(t)
        data_dict["text_pred"] = t

        if return_cos_sim:
            c = torch.cosine_similarity(x, t, dim=-1)
            data_dict["cos_sim"] = c

        return data_dict    

    
class EstLossSepEmb(torch.nn.Module):
    def __init__(self, args):
        self.args = args
        super().__init__()

    def forward(self, data_dict):
        return_dict = {}
        out_vis = data_dict["vis_pred"]
        out_text = data_dict["text_pred"]
        gt_vis = data_dict["vis_feats_proj"]
        gt_text = data_dict["caption_emb"]
        narr_vis = data_dict["vis_feats_proj_narr"]
        bb = out_vis.shape[0]

        cos_sim_vis = torch.cosine_similarity(out_vis, gt_vis, dim=-1)
        cos_sim_text = torch.cosine_similarity(out_text, gt_text, dim=-1)

        vis_loss = torch.mean(1 - cos_sim_vis)
        text_loss = torch.mean(1 - cos_sim_text)
        loss = self.args.v_lambda * vis_loss + self.args.t_lambda * text_loss
    
        return_dict["vis_loss"] = vis_loss
        return_dict["text_loss"] = text_loss
        return_dict["loss"] = loss

        # model = data_dict["model"]
        with torch.no_grad():
            # create vis to repeat
            v_r = torch.repeat_interleave(out_vis, bb, 0)
            gt_v_r = torch.repeat_interleave(gt_vis, bb, 0)
            narr_vis_r = torch.repeat_interleave(narr_vis, bb, 0)
            t_r = gt_text.repeat(out_text.shape[0], 1)
            gt_t_r = gt_text.repeat(gt_text.shape[0], 1)
            labels = torch.arange(bb).to(gt_vis.device)

            #acc of gt_vis with gt text
            comb_sims = rearrange(torch.cosine_similarity(gt_v_r, gt_t_r, dim=-1), '(v t) -> v t', v = bb)
            return_dict["gt_v-gt_t"] = torch.mean(torch.eq(torch.argmax(comb_sims, dim=-1), labels).float()) * 100
            #acc of gt text with pred vis
            comb_sims = rearrange(torch.cosine_similarity(v_r, gt_t_r, dim=-1), '(v t) -> v t', v = bb)
            return_dict["v-gt_t"] = torch.mean(torch.eq(torch.argmax(comb_sims, dim=-1), labels).float()) * 100
            #acc of gt vis with pred text
            comb_sims = rearrange(torch.cosine_similarity(gt_v_r, t_r, dim=-1), '(v t) -> v t', v = bb)
            return_dict["gt_v-t"] = torch.mean(torch.eq(torch.argmax(comb_sims, dim=-1), labels).float()) * 100
            #acc of pred_vis with pred text
            comb_sims = rearrange(torch.cosine_similarity(v_r, t_r, dim=-1), '(v t) -> v t', v = bb)
            return_dict["v-t"] = torch.mean(torch.eq(torch.argmax(comb_sims, dim=-1), labels).float()) * 100
            # acc of narr_vs with gt text
            comb_sims = rearrange(torch.cosine_similarity(narr_vis_r, gt_t_r, dim=-1), '(v t) -> v t', v = bb)
            return_dict["narr_v-gt_t"] = torch.mean(torch.eq(torch.argmax(comb_sims, dim=-1), labels).float()) * 100
            # acc of narr_vs with pred text
            comb_sims = rearrange(torch.cosine_similarity(narr_vis_r, t_r, dim=-1), '(v t) -> v t', v = bb)
            return_dict["narr_v-t"] = torch.mean(torch.eq(torch.argmax(comb_sims, dim=-1), labels).float()) * 100

        return return_dict

class TP_T4(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.combined=False
        self.args = args

        # vision estimate
        v_encoder_layer = torch.nn.TransformerEncoderLayer(d_model=768, nhead=4, batch_first=True)
        self.v_encoder = torch.nn.TransformerEncoder(v_encoder_layer, num_layers=1)
        self.v_latent = torch.nn.Parameter(torch.zeros(1,1,768))
        self.v_head = torch.nn.Linear(768, 256)

        # self.t_pe = torch.nn.Parameter(torch.zeros(1, self.args.n_img_q + 1, 768))
        self.t_pe = torch.nn.Parameter(torch.zeros(1, self.args.n_img_q + 2, 768))
        t_encoder_layer = torch.nn.TransformerEncoderLayer(d_model=768, nhead=4, batch_first=True)
        self.t_encoder = torch.nn.TransformerEncoder(t_encoder_layer, num_layers=1)
        self.t_latent = torch.nn.Parameter(torch.zeros(1,1,768))
        self.t_head = torch.nn.Linear(768, 256)

    def forward(self, data_dict, return_cos_sim=False):
        #vision estimate
        x = data_dict["vis_feats"]
        l = self.v_latent.repeat(x.shape[0], 1, 1)
        x = torch.cat([l, x], dim=1)
        x = self.v_encoder(x)
        x = x[:,0,:]
        x = self.v_head(x)
        data_dict["vis_pred"] = x

        t = data_dict["bos_feats"]
        t = rearrange(t, 'b d -> b 1 d')
        lt = self.t_latent.repeat(t.shape[0], 1, 1)

        t = torch.cat([lt, t], dim=1)
        t = torch.cat([t, data_dict["vis_feats"]], dim=1)
        t = t + self.t_pe[:, :t.shape[1], :]
        t = self.t_encoder(t)
        t = t[:,0,:]
        t = self.t_head(t)
        data_dict["text_pred"] = t


        return data_dict


class EstLossComb(torch.nn.Module):
    def __init__(self,args):
        super().__init__()
        self.loss_fn = torch.nn.MSELoss()
    def forward(self, data_dict):
        return_dict = {}
        output = data_dict["output"]
        target = data_dict["comb_sims_target"]

        # output = rearrange(output, '... (v t) -> ... v t', v = target.shape[0])
        target_singles = rearrange(target, 'v t -> (v t)')
        loss = self.loss_fn(output, target_singles)
        return_dict["loss"] = loss

        # clip accuracy
        with torch.no_grad():
            logits = rearrange(output, '(v t) -> v t', v = target.shape[0])
            labels = torch.arange(target.shape[0]).to(logits.device)

            acc = torch.mean(torch.eq(torch.argmax(logits, dim=-1), labels).float())
            return_dict["acc"] = acc * 100

            gt_acc = torch.mean(torch.eq(torch.argmax(target, dim=-1), labels).float())
            return_dict["gt_acc"] = gt_acc * 100           

        return return_dict

