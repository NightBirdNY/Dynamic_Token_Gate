import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
import random

# 1. MNIST Verisetinden Gerçek Bir Görüntü İndir/Yükle
print("MNIST veri seti yükleniyor (Sadece ilk kullanımda indirilir)...")
transform = transforms.Compose([transforms.ToTensor()])
mnist_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Rastgele bir görsel alalım
rastgele_indeks = random.randint(0, len(mnist_dataset) - 1)
image, label = mnist_dataset[rastgele_indeks]

# 2. Görüntüyü Yamalara (Patch) Bölme İşlemi
# ViT mantığı: 28x28 resmi, 4x4'lük yamalara bölelim. Toplam 7x7 = 49 yama (token) elde edeceğiz.
patch_size = 4
# Unfold ile resmi yamalara ayırıyoruz
patches = image.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
# Boyutları düzenliyoruz: [1 (Batch), 49 (Token Sayısı), 16 (4x4 Piksel)]
patches = patches.contiguous().view(1, -1, patch_size * patch_size)

# 3. Eğitilmiş Bir Kapıyı Simüle Etmek İçin Hile (Heuristic)
# Normalde burada "self.score_predictor(x)" adlı sinir ağı olurdu.
# Biz eğitilmiş bir ağın yapacağı şeyi simüle ediyoruz: "Eğer yama karanlıksa (arka plansa) logit düşük olsun, parlaksa yüksek olsun."
patch_means = patches.mean(dim=-1, keepdim=True) # Her yamanın ortalama parlaklığı
# Logitleri oluştur (2 Sınıf: [0: Sil, 1: Tut])
# Parlak yamalar için "Tut" ihtimali yüksek, "Sil" ihtimali düşük logitler veriyoruz
logits_tut = (patch_means - 0.1) * 10
logits_sil = -logits_tut
logits = torch.cat([logits_sil, logits_tut], dim=-1)

# 4. Gumbel-Softmax Kapısı (Bildiğimiz mekanizma)
tau = 0.1 # Düşük sıcaklık = Keskin kararlar (Hard Gate)
gate_decisions = F.gumbel_softmax(logits, tau=tau, hard=True)
keep_mask = gate_decisions[:, :, 1:2] # Sadece 'Tut' kararlarını al (1 veya 0)

# 5. Görselleştirme (Matplotlib)
mask_2d = keep_mask.view(7, 7).detach().numpy()
image_2d = image.squeeze().numpy()

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# A. Orijinal Görüntü
axes[0].imshow(image_2d, cmap='gray')
axes[0].set_title(f"Orijinal MNIST (Rakam: {label})")
axes[0].axis('off')

# B. Gumbel-Softmax Maskesi (Kapının Kararı)
axes[1].imshow(mask_2d, cmap='Blues', vmin=0, vmax=1)
axes[1].set_title("Kapı Ağı Kararı (7x7 Izgara)\nKoyu: Tut, Beyaz: Sil")
axes[1].axis('off')

# C. Ağın Gerçekte İşleyeceği Sonuç (Filtrelenmiş)
# Yamaları orijinal resmin üzerinde maskeliyoruz
masked_image = image_2d.copy()
for i in range(7):
    for j in range(7):
        if mask_2d[i, j] == 0: # Eğer kapı bu yamayı kapattıysa
            # O yamanın denk geldiği 4x4 piksellik alanı siyaha boya
            masked_image[i*4:(i+1)*4, j*4:(j+1)*4] = 0

axes[2].imshow(masked_image, cmap='gray')
axes[2].set_title("Transformer'ın Gördüğü\n(Arka Plan Çöpe Atıldı)")
axes[2].axis('off')

plt.tight_layout()
plt.show()