# models/dialogue_aggregator.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGATConv
import config


class ModalityQualityEstimator(nn.Module):
    def __init__(self, dim_prosody=4, dim_visual=config.VISUAL_INPUT_DIM, hidden_dim=64):
        super().__init__()
        self.audio_q_net = nn.Sequential(
            nn.Linear(dim_prosody, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  
        )

        self.visual_q_net = nn.Sequential(
            nn.Linear(dim_visual, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, prosody, visual_feat):
        p_vec = prosody.mean(dim=1) if prosody.dim() == 3 else prosody
        v_vec = visual_feat.mean(dim=1) if visual_feat.dim() == 3 else visual_feat

        alpha_a = self.audio_q_net(p_vec)
        alpha_v = self.visual_q_net(v_vec)

        alpha_t = torch.ones_like(alpha_a)

        return alpha_t, alpha_a, alpha_v


class BiDirectionalCrossAttention(nn.Module):
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.attn_t_a = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        self.attn_t_v = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)

        self.attn_a_t = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
        self.attn_v_t = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, u_t, u_a, u_v):
        if u_t.dim() == 2: u_t = u_t.unsqueeze(1)
        if u_a.dim() == 2: u_a = u_a.unsqueeze(1)
        if u_v.dim() == 2: u_v = u_v.unsqueeze(1)

        h_t_a, _ = self.attn_t_a(query=u_t, key=u_a, value=u_a)
        h_t_v, _ = self.attn_t_v(query=u_t, key=u_v, value=u_v)

        h_a_t, _ = self.attn_a_t(query=u_a, key=u_t, value=u_t)
        h_v_t, _ = self.attn_v_t(query=u_v, key=u_t, value=u_t)

        out_t = self.norm(u_t + self.dropout(h_t_a) + self.dropout(h_t_v)).squeeze(1)

        out_a = self.norm(u_a + self.dropout(h_a_t)).squeeze(1)
        out_v = self.norm(u_v + self.dropout(h_v_t)).squeeze(1)

        return out_t, out_a, out_v


class RobustFusionAggregator(nn.Module):
    def __init__(self, d_model=1024, dropout=0.3):
        super().__init__()
        self.quality_net = ModalityQualityEstimator(dim_prosody=config.PROSODY_FEATURE_DIM,
                                                    dim_visual=config.VISUAL_INPUT_DIM)
        self.bi_attn = BiDirectionalCrossAttention(d_model, dropout=dropout)

        self.router = nn.Sequential(
            nn.Linear(d_model * 3, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 3), 
            nn.Softmax(dim=-1)
        )

        self.mod_drop_prob = dropout

    def forward(self, u_t, u_a, u_v, raw_prosody, raw_visual):
        batch_size = u_t.size(0)

        q_t, q_a, q_v = self.quality_net(raw_prosody, raw_visual)

        u_a_weighted = u_a * q_a
        u_v_weighted = u_v * q_v
        u_t_weighted = u_t * q_t 

        if self.training:
            keep_prob = 1.0 - self.mod_drop_prob
            mask = torch.rand(batch_size, 3, device=u_t.device) < keep_prob

            all_dropped = (~mask).all(dim=1)
            mask[all_dropped, 0] = True  

            mask_t = mask[:, 0].unsqueeze(1)
            mask_a = mask[:, 1].unsqueeze(1)
            mask_v = mask[:, 2].unsqueeze(1)

            u_t_weighted = u_t_weighted * mask_t
            u_a_weighted = u_a_weighted * mask_a
            u_v_weighted = u_v_weighted * mask_v

        h_t, h_a, h_v = self.bi_attn(u_t_weighted, u_a_weighted, u_v_weighted)

        combined = torch.cat([h_t, h_a, h_v], dim=-1)
        weights = self.router(combined) 

        w_t = weights[:, 0].unsqueeze(1)
        w_a = weights[:, 1].unsqueeze(1)
        w_v = weights[:, 2].unsqueeze(1)

        z_fused = w_t * h_t + w_a * h_a + w_v * h_v

        return z_fused

class ContextualEncoder(nn.Module):
    def __init__(self, d_model, num_relations):
        super().__init__()
        self.gru = nn.GRU(d_model, d_model // 2, num_layers=2,
                          bidirectional=True, batch_first=True, dropout=0.2)

        self.rgat_layer1 = RGATConv(d_model, d_model, num_relations=num_relations)
        self.rgat_layer2 = RGATConv(d_model, d_model, num_relations=num_relations)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, edge_index, edge_type, batch_ptr=None):
        x_seq = x.unsqueeze(0) 
        self.gru.flatten_parameters()
        rnn_out, _ = self.gru(x_seq)
        rnn_out = rnn_out.squeeze(0) 

        x = x + rnn_out
        x = self.norm(x)

        x_gnn = self.rgat_layer1(x, edge_index, edge_type)
        x_gnn = F.relu(x_gnn)
        x_gnn = self.dropout(x_gnn)

        x_gnn = self.rgat_layer2(x_gnn, edge_index, edge_type)

        return x + x_gnn


class DialogueContextAggregator(nn.Module):
    def __init__(self, d_model=config.D_MODEL, num_relations=config.NUM_RELATIONS):
        super().__init__()

        self.fusion_layer = RobustFusionAggregator(d_model=d_model, dropout=config.MODALITY_DROPOUT_RATE)
        self.fusion_norm = nn.LayerNorm(d_model)

        self.context_encoder = ContextualEncoder(d_model, num_relations)

    def forward(self, u_t, u_a, u_v, graph_batch, raw_prosody, raw_visual):
        fused = self.fusion_layer(u_t, u_a, u_v, raw_prosody, raw_visual)
        fused = self.fusion_norm(fused)

        contextual_output = self.context_encoder(
            fused,
            graph_batch.edge_index,
            graph_batch.edge_type,
            graph_batch.ptr
        )

        return contextual_output