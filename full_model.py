# models/full_model.py

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import config
from models.unimodal_encoders import UnimodalEncodersAdvanced
from models.dialogue_aggregator import DialogueContextAggregator
from models.interpretable_bottleneck import InterpretableBottleneck
from models.classifiers import MainClassifier, FrontDoorClassifier


class ChainCRF(nn.Module):
    def __init__(self, num_tags):
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))

    def forward(self, emissions, tags, mask=None):        
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8)

        score = torch.zeros(1, device=emissions.device)

        for i in range(tags.shape[1] - 1):
            cur_tag = tags[:, i]
            next_tag = tags[:, i + 1]
            score += self.transitions[cur_tag, next_tag].sum()

        return -score / tags.shape[1]  


class CausalMDERModel(nn.Module):
    def __init__(self, num_classes=config.NUM_CLASSES, tokenizer_len=None):
        super().__init__()

        self.unimodal_encoders = UnimodalEncodersAdvanced(tokenizer_len=tokenizer_len)

        self.dialogue_aggregator = DialogueContextAggregator()

        self.interpretable_bottleneck = InterpretableBottleneck()

        self.main_classifier = MainClassifier(num_classes=num_classes)
        self.fd_classifier = FrontDoorClassifier(
            m_dim=config.M_DIM,
            x_dim=config.D_MODEL, 
            num_classes=num_classes
        )

        self.crf = ChainCRF(num_tags=num_classes)

        self.micro_batch_size = 4

    def forward(self, utterances, graph_batch, labels=None):

        num_utts = len(utterances)

        t_ids = torch.stack([u['text_ids'] for u in utterances]).to(config.DEVICE)
        t_mask = torch.stack([u['text_mask'] for u in utterances]).to(config.DEVICE)

        audio_list = [u['audio_wave'] for u in utterances]
        a_wave = pad_sequence(audio_list, batch_first=True, padding_value=0.0).to(config.DEVICE)

        prosody_list = [u['prosody_features'] for u in utterances]
        a_prosody = pad_sequence(prosody_list, batch_first=True, padding_value=0.0).to(config.DEVICE)

        visual_list = [u['openface_features'] for u in utterances]
        v_visual = pad_sequence(visual_list, batch_first=True, padding_value=0.0).to(config.DEVICE)

        v_lens = [v.shape[0] for v in visual_list]
        max_v_len = v_visual.shape[1]
        v_mask = torch.zeros((num_utts, max_v_len), dtype=torch.bool, device=config.DEVICE)
        for i, l in enumerate(v_lens):
            v_mask[i, l:] = True  

        t_feat_list, a_feat_list, v_feat_list = [], [], []

        for i in range(0, num_utts, self.micro_batch_size):
            end = min(i + self.micro_batch_size, num_utts)

            mb_t_ids = t_ids[i:end]
            mb_t_mask = t_mask[i:end]
            mb_a_wave = a_wave[i:end]
            mb_a_prosody = a_prosody[i:end]
            mb_v_visual = v_visual[i:end]

            ft, fa, fv, _, _ = self.unimodal_encoders(
                mb_t_ids, mb_t_mask,
                mb_a_wave, mb_a_prosody,
                mb_v_visual, v_mask
            )

            t_feat_list.append(ft)
            a_feat_list.append(fa)
            v_feat_list.append(fv)

        t_feat = torch.cat(t_feat_list, dim=0) 
        v_feat = torch.cat(v_feat_list, dim=0)

        u_t = t_feat[:, 0, :] 
        u_a = torch.max(a_feat, dim=1)[0]
        u_v = torch.max(v_feat, dim=1)[0]

        Z_C = self.dialogue_aggregator(
            u_t, u_a, u_v,
            graph_batch,
            a_prosody,
            v_visual
        )

        logits_main = self.main_classifier(Z_C)

        bottleneck_out = self.interpretable_bottleneck(u_t, u_a, u_v)
        M = bottleneck_out["M"]

        idx = torch.randperm(Z_C.size(0))
        Z_shuffle = Z_C[idx]

        logits_fd = self.fd_classifier(M, Z_shuffle)

        logits_obs = self.fd_classifier(M, Z_C)

        crf_loss = torch.tensor(0.0).to(config.DEVICE)
        if labels is not None:
            labels_seq = labels.unsqueeze(0)
            crf_loss = self.crf(logits_main.unsqueeze(0), labels_seq)

        return {
            "logits_main": logits_main,
            "logits_fd": logits_fd,
            "logits_obs": logits_obs,
            "bottleneck_out": bottleneck_out,
            "crf_loss": crf_loss
        }