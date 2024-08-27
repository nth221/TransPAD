import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from einops import rearrange, repeat
from einops.layers.torch import Rearrange


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        x = self.net(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        all_attn_temp = attn.mean(dim=1)

        diag_attn_return = torch.zeros((len(all_attn_temp),len(all_attn_temp[0])), dtype=torch.float32)
        for self_index in range(len(all_attn_temp[0])):
            diag_attn_return[:, self_index] = all_attn_temp[:, self_index, self_index]

        det_attn = diag_attn_return[:, 0]

        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out), det_attn.unsqueeze(dim=0), diag_attn_return.unsqueeze(dim=0)

class Transformer(nn.Module):
    def __init__(self, num_layers, encoder_dim, decoder_dim, dim, heads, dim_head, dropout = 0.):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.layers = nn.ModuleList([])

        for m_num in range(num_layers):
            self.layers.append(nn.ModuleList([
                PreNorm(encoder_dim[m_num], Attention(encoder_dim[m_num], heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(encoder_dim[m_num], FeedForward(encoder_dim[m_num], encoder_dim[m_num + 1], dropout = dropout))
            ]))
    
    def forward(self, x):
        att_tensor = torch.tensor([])
        all_att_tensor = torch.tensor([])

        for idx, (attl, ff) in enumerate(self.layers):
            z, att, all_att = attl(x)
            x = z + x

            if self.encoder_dim[idx] == self.encoder_dim[idx+1]:
                x = ff(x) + x
            else:
                x = ff(x)

            if idx == 0:
                att_tensor = att
                all_att_tensor = all_att
            else:
                att_tensor = torch.cat((att_tensor, att), dim=0)
                all_att_tensor = torch.cat((all_att_tensor, all_att), dim=0)

        att_return = att_tensor.mean(dim=0)
        all_att_return = all_att_tensor.mean(dim=0)

        return x, att_return, all_att_return


class TransPAD(nn.Module):
    def __init__(self, num_layers, encoder_dim, decoder_dim, input_dim, dim, heads, dim_head, dropout=0.1):
        super().__init__()
        self.transformer = Transformer(num_layers, encoder_dim, decoder_dim, dim, heads, dim_head, dropout)
 
        self.to_embedding = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, dim),
            nn.LayerNorm(dim)
        )

        self.to_latent = nn.Identity()

        self.decoder_layers = nn.ModuleList([])
        for m_num in range(num_layers):
            self.decoder_layers.append(nn.ModuleList([
                nn.Linear(decoder_dim[m_num], decoder_dim[m_num + 1]),
                nn.LayerNorm(decoder_dim[m_num + 1]),
                nn.GELU()
            ]))
        self.decoder_mlp_last_layer = nn.Linear(decoder_dim[num_layers], decoder_dim[num_layers])


        self.recons_embedding = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, input_dim)
        )

    def forward(self, tokens):
        x = self.to_embedding(tokens)

        b, n, _ = x.shape

        x, att, all_att = self.transformer(x)

        x = self.to_latent(x)

        for idx, (linear, norm, act) in enumerate(self.decoder_layers):
            x = linear(x)
            x = norm(x)
            x = act(x)       

        x = self.decoder_mlp_last_layer(x)        
        x = self.recons_embedding(x)

        return x, att, all_att
