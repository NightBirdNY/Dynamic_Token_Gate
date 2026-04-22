import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import seaborn as sns


# 1.KAPININ MİMARİ TANIMI-
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
        logits = self.score_predictor(x)

        if self.training:
            gate_decisions = F.gumbel_softmax(logits, tau=tau, hard=True)
        else:
            gate_decisions = F.one_hot(logits.argmax(dim=-1), num_classes=2).float()

        keep_mask = gate_decisions[:, :, 1:2]
        x_filtered = x * keep_mask

        return x_filtered, keep_mask

# 2. TEST VE ÇİZİM KISMI
if __name__ == "__main__":
    print("AIception çalışıyor: Model oluşturuluyor ve sanal veri üretiliyor...")

    batch_size = 1
    num_tokens = 196  # 14x14 izgara
    embed_dim = 64

    # Sanal video karesi
    x_dummy = torch.randn(batch_size, num_tokens, embed_dim)

    gate = DynamicTokenGate(embed_dim)
    gate.train()

    # Çıktıları alıyoruz
    _, keep_mask = gate(x_dummy, tau=0.5)

    # 1D Tensörü (196x1), resim gibi görebilmek için 2D Matrise (14x14) çeviriyoruz
    mask_2d = keep_mask[0, :, 0].detach().numpy().reshape(14, 14)

    print("Isı haritası çiziliyor")

    plt.figure(figsize=(6, 6))
    sns.heatmap(mask_2d, cmap="Blues", cbar=True, square=True,
                linewidths=0.5, linecolor='gray', vmin=0, vmax=1)

    plt.title("Gumbel-Softmax Dinamik Yama Eleme \n(1 = Koyu Mavi: Tut, 0 = Beyaz: Sil)")
    plt.axis('off')

    # Çıktı kaydetmesi için
    plt.savefig("gumbel_softmax_haritasi.png", dpi=300, bbox_inches='tight')
    print("Grafik başarıyla 'gumbel_softmax_haritasi.png' olarak kaydedildi!")

    # Pencereyi ekranda tutması için
    plt.show()