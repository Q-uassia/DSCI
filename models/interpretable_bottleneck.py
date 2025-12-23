# models/interpretable_bottleneck.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import config


class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class GradientReversal(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFn.apply(x, self.alpha)


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        distances = (torch.sum(inputs ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(inputs, self._embedding.weight.t()))

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self._embedding.weight)

        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        norm_perplexity = perplexity / self._num_embeddings

        return loss, quantized, norm_perplexity, encoding_indices


class InterpretableBottleneck(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_dim = config.D_MODEL * 3
        self.hidden_dim = config.M_DIM 

        self.compressor = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_dim * 2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )

        self.vq_layer = VectorQuantizer(
            config.VQ_NUM_EMBEDS,
            config.M_DIM,
            config.VQ_COMMITMENT_COST
        )

        self.grl = GradientReversal(alpha=1.0)
        self.speaker_discriminator = nn.Sequential(
            nn.Linear(self.hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, config.NUM_SPEAKERS)
        )

        self.aux_polarity = nn.Linear(self.hidden_dim, config.TEXT_POLARITY_DIM)
        self.aux_prosody = nn.Linear(self.hidden_dim, config.PROSODY_FEATURE_DIM)
        self.aux_visual = nn.Linear(self.hidden_dim, 10) # 假设预测 10 个关键 AU

    def forward(self, u_t, u_a, u_v):

        if u_t.dim() == 3:
            u_t = u_t.mean(dim=1)
        if u_a.dim() == 3:
            u_a = u_a.mean(dim=1)
        if u_v.dim() == 3:
            u_v = u_v.mean(dim=1)

        x_fused = torch.cat([u_t, u_a, u_v], dim=-1)

        m_raw = self.compressor(x_fused)

        vq_loss, M, perplexity, _ = self.vq_layer(m_raw)

        m_reversed = self.grl(M)
        speaker_logits = self.speaker_discriminator(m_reversed)

        pred_polarity = self.aux_polarity(M)
        pred_prosody = self.aux_prosody(M)
        pred_visual = self.aux_visual(M)

        return {
            "M": M, 
            "m_raw": m_raw, 
            "vq_loss": vq_loss,
            "perplexity": perplexity,
            "speaker_logits": speaker_logits,
            "aux_preds": {
                "polarity": pred_polarity,
                "prosody": pred_prosody,
                "visual": pred_visual
            }
        }