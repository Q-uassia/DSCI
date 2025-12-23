# models/classifiers.py

import torch
import torch.nn as nn
import config

class MainClassifier(nn.Module):
    def __init__(self, d_model=config.D_MODEL, num_classes=config.NUM_CLASSES): 
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(d_model // 2, num_classes) 
        )

    def forward(self, contextual_features):
        logits = self.classifier(contextual_features)
        return logits

class FrontDoorClassifier(nn.Module):
    def __init__(self, m_dim, x_dim, hidden_dim=config.FD_HIDDEN_DIM,
                 num_classes=config.NUM_CLASSES):
        super().__init__()
        self.classifier_head = nn.Sequential(
            nn.Linear(m_dim + x_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(hidden_dim, num_classes) 
        )

    def forward(self, m_features, x_features):
        combined_features = torch.cat([m_features, x_features], dim=-1)
        logits = self.classifier_head(combined_features)
        return logits