"""嚴格遵照 2015 年原始論文實作的 UNet，用於二元語義分割。

論文規則（Kaggle 討論區要求）：
  - 禁止 BatchNorm（原始論文沒有）
  - 禁止 padding（所有 Conv2d 使用 padding=0）
  - 上採樣必須用 ConvTranspose2d（2×2 transposed conv），不可用 bilinear
  - Skip connection 必須用 center crop，不是 padding

允許的調整（依實驗室說明）：
  - in_channels: 1 → 3（RGB 輸入，原論文是灰階）
  - out_channels: 2 → 1（單通道 sigmoid 做 binary segmentation）

參考文獻：Ronneberger et al., "U-Net: Convolutional Networks for Biomedical
Image Segmentation", MICCAI 2015. https://arxiv.org/abs/1505.04597v1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """兩層連續的 Conv3×3 → ReLU。

    嚴格遵照 2015 年論文：無 padding（unpadded）、無 BatchNorm。
    每個 block 讓空間尺寸縮小 4px（每個 3×3 conv 縮 2px）。
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            # 第一層 3×3 卷積，無 padding → 空間縮小 2px
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=0, bias=True),
            nn.ReLU(inplace=True),
            # 第二層 3×3 卷積，無 padding → 空間再縮小 2px
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=0, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    """MaxPool 2×2（stride=2）後接 ConvBlock（Encoder 的一個下採樣步驟）。"""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        # MaxPool 把空間尺寸縮小一半
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # ConvBlock 做兩層特徵提取
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))


def _center_crop(enc_feat, target_h, target_w):
    """將 Encoder 特徵圖做中心裁切，對齊 Decoder 的空間尺寸。

    論文 Figure 1 中的「copy and crop」操作。
    因為 Encoder 使用 unpadded conv，特徵圖比 Decoder 對應層大，
    concat 之前必須先裁切對齊。

    Args:
        enc_feat: Encoder 特徵圖，shape (B, C, H, W)
        target_h: 目標高度（Decoder 特徵圖的高）
        target_w: 目標寬度（Decoder 特徵圖的寬）
    Returns:
        中心裁切後的特徵圖，shape (B, C, target_h, target_w)
    """
    h, w = enc_feat.shape[2], enc_feat.shape[3]
    # 計算中心裁切的起始位置
    top  = (h - target_h) // 2
    left = (w - target_w) // 2
    return enc_feat[:, :, top:top + target_h, left:left + target_w]


class UpBlock(nn.Module):
    """Up-conv 2×2、copy-and-crop、然後 ConvBlock（Decoder 的一個上採樣步驟）。

    嚴格遵照論文流程：
      1. Up-conv 2×2（ConvTranspose2d，channel 數減半）
      2. 將對應 Encoder 特徵圖做 center crop 後 concat
      3. 兩層 3×3 Conv + ReLU（無 padding，無 BN）
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        # Up-conv 2×2：channel in_ch → in_ch//2，空間尺寸 ×2
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        # concat 後 channel 數：in_ch//2（up）+ in_ch//2（skip）= in_ch
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x, skip):
        # 先做 transposed conv 上採樣
        x = self.up(x)
        # 把 Encoder skip connection 裁切到與 x 相同大小
        skip = _center_crop(skip, x.shape[2], x.shape[3])
        # 在 channel 維度 concat（skip 在前，符合論文圖示）
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """原始 UNet（Ronneberger et al., 2015）嚴格實作版本。

    以 256×256 輸入為例的架構示意：
        Encoder（編碼器）：
            enc1:  3  → 64   ch  (ConvBlock，unpadded)
            enc2:  64 → 128  ch  (MaxPool + ConvBlock)
            enc3: 128 → 256  ch  (MaxPool + ConvBlock)
            enc4: 256 → 512  ch  (MaxPool + ConvBlock)
        Bottleneck（瓶頸層）：
            512 → 1024 ch         (MaxPool + ConvBlock)
        Decoder（解碼器）：
            dec4: 1024 → 512  ch  (UpBlock)
            dec3:  512 → 256  ch  (UpBlock)
            dec2:  256 → 128  ch  (UpBlock)
            dec1:  128 →  64  ch  (UpBlock)
        Output（輸出層）：
            64 → 1 ch  (Conv1×1 + sigmoid)

    注意：因為 unpadded conv，輸出空間尺寸比輸入小。
    forward() 末尾用 bilinear interpolation 將輸出 resize 回輸入尺寸，
    這是後處理步驟，不屬於網路架構本身。

    【本次新增】_init_weights()：
        使用 Kaiming Normal 初始化所有 Conv2d 和 ConvTranspose2d 的權重。
        沒有 BatchNorm 的情況下，好的初始化對防止 early training 中的
        activation collapse 非常重要。
    """

    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        # ---- Encoder（編碼器） ----
        # enc1 不做 MaxPool，直接對輸入做 ConvBlock
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = DownBlock(64,  128)
        self.enc3 = DownBlock(128, 256)
        self.enc4 = DownBlock(256, 512)

        # ---- Bottleneck（瓶頸層） ----
        self.bottleneck = DownBlock(512, 1024)

        # ---- Decoder（解碼器） ----
        self.dec4 = UpBlock(1024, 512)
        self.dec3 = UpBlock(512,  256)
        self.dec2 = UpBlock(256,  128)
        self.dec1 = UpBlock(128,   64)

        # ---- 輸出層 ----
        # 1×1 卷積把 64 channel 壓到 1 channel
        # （原論文是 2 channel softmax，此處改為 1 channel sigmoid 做 binary）
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

        # 【新增】Kaiming 初始化，解決無 BN 時的訓練不穩定問題
        self._init_weights()

    def _init_weights(self):
        """Kaiming（He）初始化所有卷積層。

        【本次新增的方法】
        沒有 BatchNorm 的深層網路如果使用 PyTorch 預設的隨機初始化，
        很容易在訓練初期就讓 activation 完全崩塌（全部預測背景），
        導致 Dice score 卡在 ~0。

        Kaiming Normal init（mode='fan_out', nonlinearity='relu'）
        能讓 forward pass 的 activation variance 在各層之間保持穩定，
        使梯度訊號得以有效傳遞，大幅改善早期收斂速度。

        同樣套用到 ConvTranspose2d（decoder 上採樣層）以保持一致性。
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                # Kaiming Normal：fan_out 模式適合 ReLU 激活函數
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                # bias 初始化為 0
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        input_h, input_w = x.shape[2], x.shape[3]

        # ────────────────────────────────────────────────────────────────
        # 【Overlap-Tile Strategy — 原始論文 Section 2 的核心做法】
        #
        # 問題：UNet 使用 no-padding conv，每層 3×3 conv 讓空間縮小 2px，
        #       18 層 conv（含 encoder+decoder）共吃掉邊界 94px（每側），
        #       256×256 → 68×68（只剩中央部分有預測結果）。
        #
        # 論文解法：Mirror Padding
        #   先用鏡像填充（reflect）把輸入擴大，
        #   使 UNet 的輸出剛好覆蓋原始圖片的「完整」範圍，
        #   最後 center-crop 回原始尺寸，不需要任何 bilinear resize。
        #
        # 對 256×256 輸入：
        #   pad=94 → 444×444 → UNet 輸出 260×260 → center-crop 2px → 256×256
        # ────────────────────────────────────────────────────────────────
        PAD = 94  # 論文原理：(輸入尺寸 - 輸出尺寸) / 2 = (256 - 68) / 2 = 94
        x = F.pad(x, (PAD, PAD, PAD, PAD), mode='reflect')

        # ---- Encoder 前向傳遞（輸入已擴大為 444×444）----
        e1 = self.enc1(x)         # (B, 64,   440, 440)
        e2 = self.enc2(e1)        # (B, 128,  216, 216)
        e3 = self.enc3(e2)        # (B, 256,  104, 104)
        e4 = self.enc4(e3)        # (B, 512,   48,  48)

        # ---- Bottleneck ----
        b = self.bottleneck(e4)   # (B, 1024,  20,  20)

        # ---- Decoder（含 copy-and-crop skip connections）----
        d4 = self.dec4(b,  e4)    # → (B, 512,  36,  36)
        d3 = self.dec3(d4, e3)    # → (B, 256,  68,  68)
        d2 = self.dec2(d3, e2)    # → (B, 128, 132, 132)
        d1 = self.dec1(d2, e1)    # → (B,  64, 260, 260)

        # 1×1 卷積：260×260 輸出（剛好覆蓋原始 256×256，多出 2px 邊界）
        out = self.out_conv(d1)   # (B, 1, 260, 260)

        # Center-crop 回原始輸入尺寸：裁掉兩側各 2px 多餘邊界
        oh, ow = out.shape[2], out.shape[3]
        ch = (oh - input_h) // 2
        cw = (ow - input_w) // 2
        if ch > 0 or cw > 0:
            out = out[:, :, ch:ch + input_h, cw:cw + input_w]
        elif oh < input_h or ow < input_w:
            # 保險：萬一輸出比輸入小（不應發生），才用 bilinear 補足
            out = F.interpolate(out, size=(input_h, input_w),
                                mode='bilinear', align_corners=False)

        # Sigmoid 輸出 [0, 1] 機率值
        return torch.sigmoid(out)


if __name__ == "__main__":
    # 快速測試：確認輸入輸出 shape 與參數量
    model = UNet()
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}")
