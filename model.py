import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from rope import RotaryEmbedding


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class MaskedCrossAttention(nn.Module):
    def __init__(self, n_embd, n_head, bias=False, dropout=0):
        super().__init__()
        self.q_attn = nn.Linear(n_embd, n_embd, bias=bias)
        self.kv_attn = nn.Linear(n_embd, n_embd * 2, bias=bias)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.rope = RotaryEmbedding(n_embd//n_head, learned_freq=True, cache_if_possible=False)

    def forward(self, x, x_t, y, y_t, dist=None, min_dist=0):
        B, Tx, C = x.size()
        Ty = y.size(1)

        # Apply attention projection to x for query, y for key/value
        q = self.q_attn(x)[..., :self.n_embd]
        kv = self.kv_attn(y)
        k, v = kv.split(self.n_embd, dim=2)

        q = self.rope.rotate_queries_or_keys(q.view(B, Tx, self.n_head, C // self.n_head).transpose(1, 2), x_t.unsqueeze(1)  )  # (B, nh, Tx, hs)
        k = self.rope.rotate_queries_or_keys(k.view(B, Ty, self.n_head, C // self.n_head).transpose(1, 2), y_t.unsqueeze(1)  )  # (B, nh, Ty, hs)

       
        v = v.view(B, Ty, self.n_head, C // self.n_head).transpose(1, 2)
        
        # Construct attention mask: x_t[b, i] >= y_t[b, j]
        x_t_exp = (x_t-min_dist).unsqueeze(2)  # (B, Tx, 1)
        y_t_exp = y_t.unsqueeze(1)  # (B, 1, Ty)
        time_mask = (x_t_exp >= y_t_exp)  # (B, Tx, Ty)

        if dist:
            time_mask = time_mask & (x_t_exp <= y_t_exp + dist) 

        time_mask = time_mask.float()
        
        attn_mask = time_mask.masked_fill(time_mask == 1, 0.0) \
                       .masked_fill(time_mask == 0, float('-inf'))
        
        # Expand for heads: (B, 1, Tx, Ty)
        attn_mask = attn_mask.unsqueeze(1)

        x_out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0,
            is_causal=False  # our mask handles causality
        )

        x_out = x_out.transpose(1, 2).contiguous().view(B, Tx, C)
        x_out = self.resid_dropout(self.c_proj(x_out))
        return x_out
        


class MLP(nn.Module):

    def __init__(self, n_embd, bias=False, dropout=0):
        super().__init__()
        self.c_fc    = nn.Linear(n_embd, 1 * n_embd, bias=bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(1 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class CrossAttentionBlock(nn.Module):

    def __init__(self, n_embd, n_head, bias=False, dropout=0):
        super().__init__()
        
        self.x_ln_3 = LayerNorm(n_embd, bias=bias)
        self.y_ln_1 = LayerNorm(n_embd, bias=bias)
        self.y_attn = MaskedCrossAttention(n_embd, n_head, bias=bias, dropout=dropout)
        self.y_ln_2 = LayerNorm(n_embd, bias=bias)
        self.y_mlp = MLP(n_embd, bias=bias, dropout=dropout)

    def forward(self, y, y_t, x, x_t, min_dist=0):

        y = y + self.y_attn(
            self.y_ln_1(y), y_t,
            self.x_ln_3(x), x_t,
            min_dist = min_dist
        )
        y = y + self.y_mlp(self.y_ln_2(y))
        
        return y, x



class BidirectionalCrossAttentionBlock(nn.Module):

    def __init__(self, n_embd, n_head, bias=False, dropout=0):
        super().__init__()
        self.x_ln_1 = LayerNorm(n_embd, bias=bias)
        self.y_ln_1 = LayerNorm(n_embd, bias=bias)
        
        self.x_attn = MaskedCrossAttention(n_embd, n_head, bias=bias, dropout=dropout)
        self.y_attn = MaskedCrossAttention(n_embd, n_head, bias=bias, dropout=dropout)
        
        self.x_ln_2 = LayerNorm(n_embd, bias=bias)
        self.y_ln_2 = LayerNorm(n_embd, bias=bias)
        
        self.x_mlp = MLP(n_embd, bias=bias, dropout=dropout)
        self.y_mlp = MLP(n_embd, bias=bias, dropout=dropout)

    def forward(self, y, y_t, x, x_t):
        
        x_ln = self.x_ln_1(x)
        y_ln = self.y_ln_1(y)
        
        x = x + self.x_attn(
            x_ln, x_t,
            y_ln, y_t,
        )

        y = y + self.y_attn(
            y_ln, y_t,
            x_ln, x_t,
        )
        
        x = x + self.x_mlp(self.x_ln_2(x))
        y = y + self.y_mlp(self.y_ln_2(y))
        
        return y, x


class SelfAttentionBlock(nn.Module):

    def __init__(self, n_embd, n_head, bias=False, dropout=0):
        super().__init__()
        self.x_ln_1 = LayerNorm(n_embd, bias=bias)
        self.x_attn = MaskedCrossAttention(n_embd, n_head, bias=bias, dropout=dropout)
        self.x_ln_2 = LayerNorm(n_embd, bias=bias)
        self.x_mlp = MLP(n_embd, bias=bias, dropout=dropout)


    def forward(self, x, x_t):        
        x_ln = self.x_ln_1(x)
        x = x + self.x_attn(
            x_ln, x_t, 
            x_ln, x_t
        )
        x = x + self.x_mlp(self.x_ln_2(x))
        return x




import torch
import torch.nn as nn
from torch.nn import LayerNorm

import torch
import torch.nn as nn
from torch.nn import LayerNorm

class Encoding(nn.Module):
    def __init__(self, n_embd, d=1):
        super(Encoding, self).__init__()
        self.d = d  # Number of previous points to include
        self.n_embd = n_embd

        # Update the input size of the first Linear layer to (d + d - 1) (for x and relative time differences)
        self.encoding = nn.Sequential(
            nn.Linear(d + (d - 1), n_embd * 4),  # Input size is `d + (d - 1)` now
            nn.ReLU(),
            nn.Linear(n_embd * 4, n_embd),
            nn.ReLU(),
            LayerNorm(n_embd, bias=True)
        )

    def forward(self, x_sorted, x_t_sorted):
        """
        x: Tensor of shape (batch_size, seq_len)
        x_t: Tensor of shape (batch_size, seq_len) (relative time positions, not necessarily sorted)
        """
        batch_size, seq_len = x_sorted.shape

        # Pad x and x_t by repeating the first value for the first `d-1` positions
        padded_x = torch.cat([x_sorted[:, :1].repeat(1, self.d - 1), x_sorted], dim=1)  # Shape: (batch_size, seq_len + d - 1)
        padded_x_t = torch.cat([x_t_sorted[:, :1].repeat(1, self.d - 1), x_t_sorted], dim=1)  # Shape: (batch_size, seq_len + d - 1)

        # Create the sliding windows of size `d` for x and x_t
        windows_x = padded_x.unfold(dimension=1, size=self.d, step=1)  # Shape: (batch_size, seq_len, d)
        windows_x_t = padded_x_t.unfold(dimension=1, size=self.d, step=1)  # Shape: (batch_size, seq_len, d)

        # Compute relative time differences
        # For each window, subtract the last time value in the window from the others
        relative_time_diff = windows_x_t - windows_x_t[:, :, -1:].expand_as(windows_x_t)  # Shape: (batch_size, seq_len, d)
        relative_time_diff = relative_time_diff[:, :, :-1]  # Remove the self-difference (last column)

        # Concatenate x and relative time differences
        windows = torch.cat((windows_x, relative_time_diff), dim=-1).contiguous()  # Shape: (batch_size, seq_len, d + (d - 1))

        # Pass the concatenated windows through the encoding layers
        out = self.encoding(windows)
        
        return out




class Block(nn.Module):

    def __init__(self, n_embd, n_head, bias=False, dropout=0):
        super().__init__()
        self.cgm_ins_attn = BidirectionalCrossAttentionBlock(n_embd, n_head, bias, dropout)
        self.ins_self_attn = SelfAttentionBlock(n_embd, n_head, bias, dropout)
        self.cgm_self_attn = SelfAttentionBlock(n_embd, n_head, bias, dropout)
        self.cls_ins_attn = CrossAttentionBlock(n_embd, n_head, bias, dropout)
        self.cls_cgm_attn = CrossAttentionBlock(n_embd, n_head, bias, dropout)

    def forward(self, cls, cls_t, cgm, cgm_t, ins, ins_t, pred_time=0):
        ins = self.ins_self_attn(ins, ins_t)
        cgm = self.cgm_self_attn(cgm, cgm_t)
        cgm, ins = self.cgm_ins_attn(cgm, cgm_t, ins, ins_t)
        cls, ins = self.cls_ins_attn(cls, cls_t, ins, ins_t, pred_time)
        cls, cgm = self.cls_cgm_attn(cls, cls_t, cgm, cgm_t, pred_time)
        
        return cls, cgm, ins



class CGMPredictor(nn.Module):

    def __init__(self, 
                     n_embd = 64, 
                     n_head = 8, 
                     n_layer = 4,
                     dropout = 0.0
                 ):

        super(CGMPredictor, self).__init__()
        
        self.basal_embd = Encoding(n_embd, d=6)
        self.bolus_embd = Encoding(n_embd, d=6)
        self.cgm_embd = Encoding(n_embd, d=6)


        self.blocks = nn.ModuleList([
            Block(n_embd, n_head, bias=True, dropout=dropout) 
            for _ in range(n_layer)
        ])

        self.layernorm = LayerNorm(n_embd, bias=False)
        self.head = nn.Linear(n_embd, 1, bias=True)


        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * n_layer))

        
        
        self.basal_norm = (0, 1.5)
        self.bolus_norm = (0, 2)
        self.cgm_norm = (150, 50)
        

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            

    def normalize_basal(self, basal):
        return (basal - self.basal_norm[0])/self.basal_norm[1]
    def normalize_bolus(self, bolus):
        return (bolus - self.bolus_norm[0])/self.bolus_norm[1]
    def normalize_cgm(self, cgm):
        return (cgm - self.cgm_norm[0])/self.cgm_norm[1]
        

    def unnormalize_basal(self, basal):
        return (basal * self.basal_norm[1]) + self.basal_norm[0]
    def unnormalize_bolus(self, bolus):
        return (bolus * self.bolus_norm[1]) + self.bolus_norm[0]
    def unnormalize_cgm(self, cgm):
        return (cgm * self.cgm_norm[1]) + self.cgm_norm[0]


    
    def forward(self, 
                inp_cgm, inp_basal, inp_bolus, 
                cgm_t, basal_t, bolus_t,
                cls_t, pred_time=0):
    
        # Normalize inputs
        inp_cgm = self.normalize_cgm(inp_cgm)
        inp_basal = self.normalize_basal(inp_basal)
        inp_bolus = self.normalize_bolus(inp_bolus)
        
        
        cgm = self.cgm_embd(inp_cgm, cgm_t)
        
        if inp_basal.size(1) > 0:
            basal = self.basal_embd(inp_basal, basal_t)
        else:
            basal = torch.empty(inp_basal.size(0), 0, cgm.shape[-1]).to(inp_cgm.device) 
        
        if inp_bolus.size(1) > 0:
            bolus = self.bolus_embd(inp_bolus, bolus_t)
        else:
            bolus = torch.empty(inp_bolus.size(0), 0, cgm.shape[-1]).to(inp_cgm.device) 
        
        B, T = cls_t.shape
        cls = torch.zeros(B, T, cgm.shape[-1]).to(inp_cgm.device)
        
        ins = torch.cat([basal, bolus], dim=1)
        ins_t = torch.cat([basal_t, bolus_t], dim=1)
        
        # Pass through blocks
        for block in self.blocks:
            cls, cgm, ins = block(
                cls, cls_t, 
                cgm, cgm_t, 
                ins, ins_t, 
                pred_time
            )
            
            
        out = self.layernorm(cls)
        out = self.head(cls)
        

        
        ref_time = cls_t - pred_time  # (batch, n_times)

        cgm_time_exp = cgm_t.unsqueeze(1)  # (batch, 1, n_cgm_times)
        ref_time_exp = ref_time.unsqueeze(-1)  # (batch, n_times, 1)
        
        mask = (cgm_time_exp <= ref_time_exp)  # (batch, n_times, n_cgm_times)
        masked_cgm_time = cgm_time_exp.masked_fill(~mask, float('-inf'))
        idx = masked_cgm_time.argmax(dim=-1)  # (batch, n_times)
        
        cgm_exp = inp_cgm.unsqueeze(1).expand(-1, idx.shape[1], -1)  # (batch, n_times, n_cgm_times)
        recent_cgm = cgm_exp.gather(2, idx.unsqueeze(-1)).squeeze(-1)  # (batch, n_times)
        
        
        out = out.squeeze(-1) + recent_cgm  # (batch, n_times)

        return self.unnormalize_cgm(out)

    
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
