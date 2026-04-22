import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicTokenGate(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.score_predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 2)
        )

    def forward(self, x, tau=1.0):
        # 1. Ham Kararların Hesaplanması
        logits = self.score_predictor(x)

        # 2. Gumbel-Softmax ve STE (Straight-Through Estimator) Yöntemi
        if self.training:
            gate_decisions = F.gumbel_softmax(logits, tau=tau, hard=True)
        else:
            gate_decisions = F.one_hot(logits.argmax(dim=-1), num_classes=2).float()

        # 3. Maskenin Oluşturulması ve Uygulanması
        keep_mask = gate_decisions[:, :, 1:2]
        x_filtered = x * keep_mask

        return x_filtered, keep_mask