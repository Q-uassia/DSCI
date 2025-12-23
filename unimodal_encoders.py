# models/unimodal_encoders.py

import torch
import torch.nn as nn
import math
from transformers import AutoModel, AutoConfig, WavLMModel, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
import torch.nn.functional as F
import config


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe.requires_grad = False
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


def get_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )


class TextEncoder(nn.Module):
    def __init__(self, pretrained_model=config.TEXT_PRETRAINED, d_model=config.D_MODEL, tokenizer_len=None):
        super().__init__()
        print(f"Loading DeBERTa from {pretrained_model}...")

        self.roberta = AutoModel.from_pretrained(pretrained_model, torch_dtype=torch.bfloat16)
        if tokenizer_len is not None:
            self.roberta.resize_token_embeddings(tokenizer_len)

        lora_config = LoraConfig(
            r=32, lora_alpha=16,
            target_modules=["query_proj", "value_proj"], 
            lora_dropout=0.05, bias="none"
        )
        self.roberta = get_peft_model(self.roberta, lora_config)
        self.projection = nn.Linear(1024, d_model, dtype=torch.bfloat16)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        return self.projection(outputs.last_hidden_state)


class AudioEncoder(nn.Module):
    def __init__(self, pretrained_model=config.AUDIO_PRETRAINED, prosody_dim=4, d_model=1024):
        super().__init__()
        print("Loading Audio Frozen + TCN...")

        self.wavlm = WavLMModel.from_pretrained(pretrained_model, torch_dtype=torch.bfloat16)
        self.wavlm.eval()
        for param in self.wavlm.parameters():
            param.requires_grad = False

        self.tcn = nn.Sequential(
            nn.Conv1d(768, d_model, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, d_model),
            nn.ReLU(),

            nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, d_model),
            nn.ReLU()
        )

        self.fusion_projection = nn.Linear(d_model + prosody_dim, d_model)

    def forward(self, input_values, prosodic_features):

        with torch.no_grad():
            if self.wavlm.dtype == torch.bfloat16 and input_values.dtype != torch.bfloat16:
                wavlm_input = input_values.to(torch.bfloat16)
            else:
                wavlm_input = input_values

            outputs = self.wavlm(wavlm_input)
            x = outputs.last_hidden_state

        x = x.to(self.tcn[0].weight.dtype)
        prosodic_features = prosodic_features.to(x.dtype)

        x = x.transpose(1, 2)
        x = self.tcn(x)
        x = x.transpose(1, 2)

        T_new = x.shape[1]

        p = prosodic_features.transpose(1, 2)
        p = F.interpolate(p, size=T_new, mode='linear', align_corners=False)
        p = p.transpose(1, 2)

        return self.fusion_projection(torch.cat([x, p], dim=-1))


class VisualEncoder(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, num_layers, dropout):
        super().__init__()
        self.input_ln = nn.LayerNorm(input_dim, dtype=torch.bfloat16)
        self.project = nn.Sequential(
            nn.Linear(input_dim, d_model, dtype=torch.bfloat16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model, dtype=torch.bfloat16)
        )

    def forward(self, x, mask):
        x = x.to(torch.bfloat16)
        x = self.input_ln(x)
        return self.project(x)


class UnimodalEncodersAdvanced(nn.Module):
    def __init__(self, tokenizer_len=None):
        super().__init__()
        self.text_encoder = TextEncoder(tokenizer_len=tokenizer_len)
        self.audio_encoder = AudioEncoder()
        self.visual_encoder = VisualEncoder(
            input_dim=config.VISUAL_INPUT_DIM,
            d_model=config.D_MODEL,
            n_heads=4, num_layers=2, dropout=config.DROPOUT
        )

    def forward(self, text_input_ids, text_attention_mask,
                audio_input_values, audio_prosody_features,
                visual_feature_sequence, visual_padding_mask):
        t_feat = self.text_encoder(text_input_ids, text_attention_mask)
        a_feat = self.audio_encoder(audio_input_values, audio_prosody_features)
        v_feat = self.visual_encoder(visual_feature_sequence, visual_padding_mask)

        r_a = torch.ones(t_feat.size(0), 1, device=t_feat.device)
        r_v = torch.ones(t_feat.size(0), 1, device=t_feat.device)

        return t_feat, a_feat, v_feat, r_a, r_v