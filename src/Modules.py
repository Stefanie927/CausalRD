import copy
import torch
import torch.nn as nn
from torch import einsum
from pathlib import Path
import math
from tqdm import tqdm
from torch.optim import Adam, SGD
import numpy as np
from torch.utils import data
from einops import rearrange, repeat
from util.utils import make_beta_schedule, default, exists, extract_into_tensor, BatchedOperation, noise_like
from util.utils import create_activation, create_norm, mean_flat
import torch.nn.functional as F
from typing import Optional
from functools import partial
from models_vit import VisionTransformer
try:
    from apex import amp
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import logging


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def max_neg_value(t):
    return -torch.finfo(t.dtype).max

def cycle(dl):
    while True:
        for data in dl:
            yield data

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def loss_backwards(fp16, loss, optimizer, **kwargs):
    if fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(**kwargs)
    else:
        loss.backward(**kwargs)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))
    
class MaskLayer(nn.Module):
    def __init__(self, num_factor, out_dim):
        super().__init__()
        self.num_factor = num_factor
        self.out_dim = out_dim
        self.mix_linear = nn.ModuleList()
        for idx in range(self.num_factor):
            self.mix_linear.append(
                nn.Sequential(
                    nn.Linear(self.out_dim, 32),
                    nn.ELU(),
                    nn.Linear(32, self.out_dim),
                )
            )

    def mix(self, z):  # z.shape(bs, concept_num, concept_dim)
        zy = z.view(-1, self.num_factor*self.out_dim)    # zy.shape(bs, concept_num*concept_dim)
        split_zy = list(torch.split(zy, zy.shape[-1]//self.num_factor, dim = 1))
        output_zy = []
        for idx, linear in enumerate(self.mix_linear):
            output_zy.append(linear(split_zy[idx]))  
        outputs = torch.cat(output_zy, dim=1)
        return outputs  # outputs.shape(bs, concept_num*concept_dim)


def kl_normal(qm, qv, pm, pv):
	"""
	Computes the elem-wise KL divergence between two normal distributions KL(q || p) and
	sum over the last dimension

	Args:
		qm: tensor: (batch, dim): q mean
		qv: tensor: (batch, dim): q variance
		pm: tensor: (batch, dim): p mean
		pv: tensor: (batch, dim): p variance

	Return:
		kl: tensor: (batch,): kl between each sample
	"""
	element_wise = 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1)
	kl = element_wise.sum(-1)
	#print("log var1", qv)
	return kl

def condition_prior(scale, label, dim):
	mean = torch.ones(label.size()[0],label.size()[1], dim)
	var = torch.ones(label.size()[0],label.size()[1], dim)
	for i in range(label.size()[0]):
		for j in range(label.size()[1]):
			mul = (float(label[i][j])-scale[j][0])/(scale[j][1]-0)
			mean[i][j] = torch.ones(dim)*mul
			var[i][j] = torch.ones(dim)*1
	return mean, var


class DisentanglementEncoder(nn.Module):
    def __init__(self, 
                 profile_size,  
                 out_dim,  # 32
                 num_factor, # len(concepts)+1
                 causal_dag,
                 label_categories,
                 device,
                 bias = False,
                 out_act = "gelu",  
                 gamma = 35
                 ):
        super().__init__()
        if isinstance(out_act, str) or out_act is None:
            out_act = create_activation(out_act)
        self.device = device
        self.num_factor = num_factor
        self.out_dim = out_dim
        self.profile_size = profile_size
        #### out_dim=1024
        self.exogenous_encoder_m_v = nn.Sequential(     
            nn.Linear(profile_size, profile_size // 4),   
            Mish(), 
            nn.Linear(profile_size // 4, num_factor * out_dim * 2)  
        )
        #### out_dim=64
        # self.exogenous_encoder_m_v = nn.Sequential(       
        #     nn.Linear(profile_size, profile_size // 2), 
        #     Mish(), 
        #     nn.Linear(profile_size // 2, num_factor * out_dim * 2)  
        # )
        # ### out_dim=2048
        # self.exogenous_encoder_m_v = nn.Sequential(      
        #     nn.Linear(profile_size, profile_size), 
        #     Mish(), 
        #     nn.Linear(profile_size, num_factor * out_dim * 2)  
        # )
        
        self.causal_dag = nn.Parameter(causal_dag)
        self.causal_dag.requires_grad = False
        self.I = nn.Parameter(torch.eye(num_factor))
        self.I.requires_grad = False
        # self.A = nn.Parameter(causal_dag)    
        self.A = nn.Parameter(torch.zeros(causal_dag.shape[0],causal_dag.shape[1]))    
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_factor))
        else:
            self.register_parameter('bias', None)

        self.label_predictor = nn.ModuleList()
        for idx, num in enumerate(label_categories):  
            self.label_predictor.append(nn.Sequential(
                nn.Linear(out_dim, num),
                nn.Softmax(dim = 1) 
            )
            )

        self.multilabelmulticate_loss = nn.CrossEntropyLoss()

        self.discriminator_ov = nn.Linear(out_dim, 1)
        self.discriminator_ov2 = nn.Linear(num_factor, 1)
        self.discriminator_ov_act = nn.Sigmoid()
        
        self.gamma = gamma

        self.mix_z = MaskLayer(self.num_factor, self.out_dim)
        self.mix_u = MaskLayer(self.num_factor-1, 1)
        self.attn = Attention(self.out_dim)

        self.scale = np.array([
            [2, 2],   
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
        ])  
        self.mse_loss = torch.nn.MSELoss()

    
    def mask_z(self, x):

        x = torch.matmul((self.causal_dag*self.A), x)
        # x = torch.matmul((self.A), x)
        
        return x
    
    def mask_u(self, x):
        x = x.view(-1, x.size()[1], 1)   # x.shape(bs,num_factor,1)
        x = torch.matmul((self.causal_dag[:-1,:-1]*self.A[:-1,:-1]), x.float())
        # x = torch.matmul((self.A[:-1,:-1]), x.float())
        
        return x
    
    def normal_kl(self, mean1, logvar1, mean2, logvar2):
        """
        Compute the KL divergence between two gaussians.

        Shapes are automatically broadcasted, so batches can be compared to
        scalars, among other use cases.
        """
        tensor = None
        for obj in (mean1, logvar1, mean2, logvar2):
            if isinstance(obj, torch.Tensor):
                tensor = obj
                break
        assert tensor is not None, "at least one argument must be a Tensor"

        # Force variances to be Tensors. Broadcasting helps convert scalars to
        # Tensors, but it does not work for th.exp().
        logvar1, logvar2 = [
            x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
            for x in (logvar1, logvar2)
        ]

        return 0.5 * (
            -1.0
            + logvar2
            - logvar1
            + torch.exp(logvar1 - logvar2)
            + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
        )

    def calculat_prior_kl(self, mean, log_var):  
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.
        """
        batch_size = mean.shape[0]
        kl_prior = self.normal_kl(
            mean1=mean, logvar1=log_var, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)
        # return sum_flat(kl_prior) / np.log(2.0)

    def sample(self, mean, log_var):
        noise = torch.randn_like(mean)
        return mean + (0.5 * log_var).exp() * noise
    
    def conditional_sample_gaussian(self, m, v):
        sample = torch.randn(m.size()).to(self.device)
        z = m + (v**0.5)*sample
        return z
    
    def calcuated_dag(self, x):
        if x.dim()>2:
            x = x.permute(0,2,1)
        x = F.linear(x, torch.inverse(self.I-(self.A*self.causal_dag)), self.bias)  
        # x = F.linear(x, torch.inverse(self.I-(self.A)), self.bias)  

        if x.dim()>2:
            x = x.permute(0,2,1).contiguous()
        return x

    
    def forward(self, x, o, lambdav=0.001): # x:bs*emb o:concept label
        exogenous_factor_m, exogenous_factor_v = torch.split(self.exogenous_encoder_m_v(x), self.num_factor * self.out_dim, dim=-1)  
        prior_kl = self.calculat_prior_kl(exogenous_factor_m, exogenous_factor_v).mean()
        
        exogenous_factor_m = rearrange(exogenous_factor_m, 'b (h d) -> b h d', h=self.num_factor)
        exogenous_factor_v = rearrange(exogenous_factor_v, 'b (h d) -> b h d', h=self.num_factor)
        # z_m = self.calcuated_dag(exogenous_factor_m)
        z_m = torch.inverse(self.I - (self.causal_dag*self.A)).matmul(exogenous_factor_m)
        # z_m = torch.inverse(self.I - (self.A)).matmul(exogenous_factor_m)
        z_v = torch.ones(z_m.shape).to(self.device)
        # concept_embs = self.sample(z_m, exogenous_factor_v)   
        concept_embs = z_m

        mask_concept_embs_m = self.mask_z(z_m)
        mask_concept_embs_v = torch.ones(mask_concept_embs_m.shape).to(self.device)
        mask_label = self.mask_u(o)
        
        feat_z = self.mix_z.mix(mask_concept_embs_m).reshape([mask_concept_embs_m.size()[0], self.num_factor, self.out_dim])
        e_tilde = self.attn.attention(z_m, exogenous_factor_m)[0]
        # m_concept_embs = m_concept_embs + exogenous_embs
        feat_z = feat_z + e_tilde
        g_u = self.mix_u.mix(mask_label)

        mask_recon_loss = ((z_m - feat_z) ** 2).mean()  
        
        cp_m, _ = condition_prior(self.scale, o, self.out_dim)  # cp_m.shape(bs,concept_num,out_dim) 
        cp_m = cp_m.to(self.device)
        feat_kl = 0
        for i in range(self.num_factor-1): 
            feat_kl = feat_kl + kl_normal(z_m[:,i,:], z_v[:,i,:], cp_m[:,i,:], z_v[:,i,:])
        feat_kl = torch.mean(feat_kl)

        mask_kl = 0
        for i in range(self.num_factor-1):
            mask_kl = mask_kl + kl_normal(feat_z[:,i,:], z_v[:,i,:], cp_m[:,i,:], z_v[:,i,:])
        mask_kl = torch.mean(mask_kl)

        label_mse = self.mse_loss(g_u, o.float())

        pred_o = []
        for idx, predictor in enumerate(self.label_predictor):
            pred_o.append(predictor(concept_embs[:,idx,:]))  # concept_embs[:,idx,:].shape=(bs,out_dim) 
        pred_o_loss = 0
        for idx, pred_o_idx in enumerate(pred_o):
            pred_o_loss_idx = self.multilabelmulticate_loss(pred_o_idx, o[:,idx])
            pred_o_loss += pred_o_loss_idx
        
        # take mean-level loss
        pred_o_loss /= idx + 1   
        
        # new adversirial part
        pred_u = []
        for idx, predictor in enumerate(self.label_predictor):
            pred_u.append(predictor(concept_embs[:, -1, :]))   
        pred_u_loss = 0
        for idx, pred_u_idx in enumerate(pred_u):
            pred_u_loss_idx = self.multilabelmulticate_loss(pred_u_idx, o[:,idx]) 
            pred_u_loss += pred_u_loss_idx
        # take mean-level loss
        discriminator_loss = - pred_u_loss / (idx + 1)  
        
        # take sum-level loss
        # discriminator_loss = - pred_u_loss

        output_age = pred_o[0]
        output_sex = pred_o[1][:,-1]   
        output = pred_o[2:]   
        output = torch.stack(output)
        output = output[:, :, -1].T

        return concept_embs, output, output_age, output_sex, mask_recon_loss, pred_o_loss, discriminator_loss, prior_kl, feat_kl.to(torch.float), mask_kl.to(torch.float), label_mse.to(torch.float)
    
    def extract_exogenous_embs(self, x):
        with torch.no_grad():
            exogenous_factor_m, exogenous_factor_v = torch.split(self.exogenous_encoder_m_v.eval()(x), self.num_factor * self.out_dim, dim=-1)
            exogenous_factor = self.sample(exogenous_factor_m, exogenous_factor_v)
            exogenous_embs = rearrange(exogenous_factor, 'b (h d) -> b h d', h=self.num_factor)
        return exogenous_embs

class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)
    
class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)

class CrossAttention(nn.Module):
    def __init__(self,
                 query_dim, 
                 context_dim, 
                 heads = 8, 
                 dim_head = 64, 
                 dropout = 0., 
                 qkv_bias = False):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        
        self.scale = dim_head ** -0.5
        self.heads = heads
        
        self.to_q = nn.Linear(query_dim, inner_dim, bias = qkv_bias)
        self.to_k = nn.Linear(context_dim, inner_dim, bias = qkv_bias)
        self.to_v = nn.Linear(context_dim, inner_dim, bias = qkv_bias)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, *, context = None, mask = None):
        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)
        
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        
        if exists(mask):
            mnv = max_neg_value(sim) - torch.finfo(sim.dtype).max
            if sim.shape[1:] == sim.shape[1:]:
                mask = repeat(mask, 'b ... -> (b h) ...', h = h)
            else:
                mask = rearrange(mask, 'b ... -> b (...)')
                mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, mnv)
        
        attn = sim.softmax(dim = -1)
        # print(attn)
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int, 
        d_head: int = 64, 
        self_attn: bool = False,
        cross_attn: bool = True,
        ts_cross_attn: bool = False, 
        final_act: Optional[nn.Module] = None,
        dropout: float = 0, 
        context_dim: Optional[int] = None, 
        gated_ff: bool = True, 
        checkpoint: bool = False,
        qkv_bias: bool = False, 
        linear_attn: bool = False, 
    ):
        super().__init__()
        assert self_attn or cross_attn, 'At least on attention layer'
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.ff = FeedForward(dim, dropout=dropout, glu = gated_ff)
        if ts_cross_attn:
            raise NotImplementedError("Deprecated, please remove.")  # FIX: remove ts_cross_attn option
        else:
            assert not linear_attn, "Performer attention not setup yet."  # FIX: remove linear_attn option
            attn_cls = CrossAttention
        
        if self.cross_attn:
            self.attn1 = attn_cls(
                query_dim = dim, 
                context_dim = context_dim, 
                heads = n_heads, 
                dim_head = d_head, 
                dropout = dropout, 
                qkv_bias = qkv_bias
            )
        if self.self_attn:
            self.attn2 = attn_cls(
                query_dim = dim, 
                heads = n_heads, 
                dim_head = d_head, 
                dropout = dropout, 
                qkv_bias = qkv_bias
            )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.act = final_act
        self.checkpoint = checkpoint
        assert not self.checkpoint, "Checkpointing not available yet"
    
    @BatchedOperation(batch_dim=0, plain_num_dim=2)
    def forward(self, x, context=None, cross_mask=None, self_mask=None, **kwargs):
        if self.cross_attn:
            x = self.attn1(self.norm1(x), context=context, mask=cross_mask, **kwargs) + x
        if self.self_attn:
            x = self.attn2(self.norm2(x), mask=self_mask, **kwargs) + x
        x = self.ff(self.norm3(x)) + x
        if self.act is not None:
            x = self.act(x)
        return x


class Attention(nn.Module):
  def __init__(self, in_features, bias=False):
    super().__init__()
    self.M =  nn.Parameter(torch.nn.init.normal_(torch.zeros(in_features,in_features), mean=0, std=1))
    self.sigmd = torch.nn.Sigmoid()
    #self.M =  nn.Parameter(torch.zeros(in_features,in_features))
    #self.A = torch.zeros(in_features,in_features).to(device)
    
  def attention(self, z, e):
    a = z.matmul(self.M).matmul(e.permute(0,2,1))
    a = self.sigmd(a)
    #print(self.M)
    A = torch.softmax(a, dim = 1)
    e = torch.matmul(A,e)
    return e, A



class Denoise_net(nn.Module):
    def __init__(self, 
                 dim, 
                 out_dim, 
                 num_factor,  # len(concepts)+1
                 causal_dag, 
                 label_categories, 
                 device,
                 depth = 4,
                 num_heads = 4, 
                 dim_head = 64,
                 dropout = 0., 
                 norm_type = "layernorm", 
                 num_layers = 1, 
                 act = 'gelu', 
                 out_act = None, 
                 with_time_emb = True):
        super().__init__()
        if isinstance(act, str) or act is None:
            act = create_activation(act)
        if isinstance(out_act, str) or out_act is None:
            out_act = create_activation(out_act)
        
        # if with_time_emb:
        #     time_dim = dim
        #     self.time_mlp = nn.Sequential(
        #         SinusoidalPosEmb(dim), 
        #         nn.Linear(dim, dim * 4), 
        #         Mish(),
        #         nn.Linear(dim * 4, dim)
        #     )
        # else:
        #     time_dim = None
        #     self.time_mlp = None
        
        # self.layers = nn.ModuleList()
        # for _ in range(num_layers - 1):
        #     self.layers.append(nn.Sequential(
        #         nn.Linear(dim, dim),
        #         act,
        #         create_norm(norm_type, dim),
        #         nn.Dropout(dropout)
        #     ))
        # self.layers.append(nn.Sequential(nn.Linear(dim, out_dim), out_act))
        
        # disentanglement encoder
        # self.DisentanglementEncoder = DisentanglementEncoder(dim, 1024, num_factor, causal_dag, label_categories, device)   # 32  1024  64
        self.DisentanglementEncoder = DisentanglementEncoder(dim, 256, num_factor, causal_dag, label_categories, device)   # 32  1024  64
        # self.Cross_attention_module = nn.ModuleList([
        #     BasicTransformerBlock(out_dim, num_heads, dim_head, self_attn=False, cross_attn=True, context_dim=32, 
        #                           qkv_bias=True, dropout=dropout, final_act=None)
        #     for _ in range(depth)
        # ])
        # self.decoder_norm = create_norm(norm_type, out_dim)

        
    def forward(self, image_feat, labels):  
        concept_embs, output, output_age, output_sex, mask_recon_loss, pred_o_loss, discriminator_loss, prior_kl, feat_kl, mask_kl, label_mse = self.DisentanglementEncoder(image_feat, labels)

        return output, output_age, output_sex, mask_recon_loss, pred_o_loss, discriminator_loss, prior_kl, feat_kl, mask_kl, label_mse
    
    def test_forward(self, image_feat, labels): 
        left_image_feat = image_feat[:, :1024]
        right_image_feat = image_feat[:, 1024:]
        
        left_concept_embs, left_output, left_output_age, left_output_sex, mask_recon_loss, pred_o_loss, discriminator_loss, prior_kl, feat_kl, mask_kl, label_mse = self.DisentanglementEncoder(left_image_feat, labels)
        right_concept_embs, right_output, right_output_age, right_output_sex, mask_recon_loss, pred_o_loss, discriminator_loss, prior_kl, feat_kl, mask_kl, label_mse = self.DisentanglementEncoder(right_image_feat, labels)

        output = torch.max(left_output, right_output)
        output_age = torch.max(left_output_age, right_output_age)
        output_sex = torch.max(left_output_sex, right_output_sex)

        return output, output_age, output_sex
    
    
    def get_features(self, image_feat, labels):  
        left_image_feat = image_feat[:, :1024]
        right_image_feat = image_feat[:, 1024:]
        
        left_concept_embs, left_output, left_output_age, left_output_sex, mask_recon_loss, pred_o_loss, discriminator_loss, prior_kl, feat_kl, mask_kl, label_mse = self.DisentanglementEncoder(left_image_feat, labels)
        right_concept_embs, right_output, right_output_age, right_output_sex, mask_recon_loss, pred_o_loss, discriminator_loss, prior_kl, feat_kl, mask_kl, label_mse = self.DisentanglementEncoder(right_image_feat, labels)

        return left_concept_embs, right_concept_embs
    
    def get_single_features(self, image_feat, labels):
        concept_embs, output, output_age, output_sex, mask_recon_loss, pred_o_loss, discriminator_loss, prior_kl, feat_kl, mask_kl, label_mse = self.DisentanglementEncoder(image_feat, labels)
        return concept_embs
    

    def no_weight_decay(self):
        no_decay = {'bias', 'LayerNorm.weight'}
        return [param for name, param in self.named_parameters() if not any(nd in name for nd in no_decay)]


class GaussianDiffusion(nn.Module):
    def __init__(self, 
                 denosie_fn, # Denoise_net
                 *, 
                 profile_size,  
                #  channels = 3, 
                 timesteps = 1000, 
                 loss_type = "l1", 
                 betas = None):
        super().__init__()
        self.profile_size = profile_size
        self.denosie_fn = denosie_fn
        
     
        
        if exists(betas):
            betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        else:
            betas = make_beta_schedule("linear", timesteps)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)   
          
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

 
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        self.loss_type = loss_type
        
        to_torch = partial(torch.tensor, dtype=torch.float32)
        
        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))
        
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))
        
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))
    
    def q_mean_variance(self, x_start, t):
        """
        Given x_0 and t, output x_t by adding noise
        """
        mean = extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract_into_tensor(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance
    
    def predict_start_from_noise(self, x_t, t, noise):
        """
        
        """
        assert x_t.shape == noise.shape, "Please check the code and data"
        return (extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - 
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
                )
    
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start + 
            extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def p_mean_variance(self, x, t, clip_denoised: bool):
        # x_recon = self.predict_start_from_noise(x, t=t, noise=self.denosie_fn(x, t))
        x_recon = self.denosie_fn(x, t)
        # this should be setted as the data distribution
        if clip_denoised:
            x_recon.clamp_(0)
            # x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance
    
    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=False, repeat_noise = False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # noise = default(noise, lambda: torch.randn_like(x_start))
        
        # no noise when t==0
        nonzero_mask = (1 - (t==0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
    
    @torch.no_grad()
    def p_sample_loop(self, shape):
        device = self.betas.device
        
        b = shape[0]
        img = torch.randn(shape, device = device)
        
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))
        return img
    
    @torch.no_grad()
    def sample(self, batch_size=16):
        profile_size = self.profile_size
        return self.p_sample_loop((batch_size, profile_size))

    def p_mean_variance_with_factor(self, x, t, concept_embs, clip_denoised: bool, eps = False):
        
        x_start = None
        if eps:
            x_recon = self.predict_start_from_noise(x, t=t, noise=self.denosie_fn(x, x_start, t, concept_embs = concept_embs))
        else:
            x_recon = self.denosie_fn(x, x_start, t, concept_embs = concept_embs)
            
        # this should be setted as the data distribution
        if clip_denoised:
            x_recon.clamp_(0)
            # x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample_with_factor(self, x, t, concept_embs, clip_denoised=False, repeat_noise = False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance_with_factor(x=x, t=t, concept_embs=concept_embs, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # noise = default(noise, lambda: torch.randn_like(x_start))
        
        # no noise when t==0
        nonzero_mask = (1 - (t==0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop_with_factor(self, shape, concept_embs):
        device = self.betas.device
        
        b = shape[0]  # shape=(bs,dim)
        img = torch.randn(shape, device = device)  # img.shape=(bs,dim)
        
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            img = self.p_sample_with_factor(img, torch.full((b,), i, device=device, dtype=torch.long), concept_embs)
        return img

    @torch.no_grad()
    def sample_with_factor(self, concept_embs, batch_size=16):
        profile_size = self.profile_size
        return self.p_sample_loop_with_factor((batch_size, profile_size), concept_embs)
    
    @torch.no_grad()
    def interpolate(self, x1, x2, t=None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)
        
        assert x1.shape == x2.shape
        
        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))
        
        img = (1 - lam) * xt1 + lam *xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total = t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))
        
        return img

    def q_sample(self, x_start, t, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +  
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
                )
            
    def p_losses(self, x_start, t, labels, weights, noise = None, eps = False):
        b, c = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))
        
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)  
        x_recon, mask_recon_loss, pred_o_loss, discriminator_loss, prior_kl = self.denosie_fn(x_noisy, x_start, t, labels)
        
        assert x_recon.shape == x_noisy.shape, "Please check the code and data"
        
        if self.loss_type == "l1":
            if eps:
                loss = (((noise - x_recon).abs()) * weights[:, None]).sum()
            else:
                loss = (((x_start - x_recon).abs()) * weights[:, None]).sum()
        elif self.loss_type == "l2":
            if eps:
                loss = (((noise - x_recon)**2) * weights[:, None]).sum()
            else:
                loss = (((x_start - x_recon)**2) * weights[:, None]).sum()
        else:
            raise NotImplementedError()
        
        return loss, mask_recon_loss, pred_o_loss, discriminator_loss, prior_kl
    
    def forward(self, x, *args, **kwargs):
        b, c, device, profile_size, = *x.shape, x.device, self.profile_size  # b=batchsize c=1000
        assert c == profile_size, f'dimension of gene expression profile must be {profile_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x, t, *args, **kwargs)








