"""ResNet-34 Encoder + UNet-style Decoder，用於二元語義分割。

Encoder（ResNet-34）從零開始訓練，遵照：
    He et al., "Deep Residual Learning for Image Recognition", CVPR 2016.
    https://arxiv.org/abs/1512.03385

Decoder 採用 UNet 風格的跳躍連接架構，參考：
    https://www.researchgate.net/publication/359463249

注意：不載入任何預訓練權重，全部從頭訓練。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# ResNet-34 基本建構單元
# ---------------------------------------------------------------------------

class BasicBlock(nn.Module):
    """ResNet-34 的基本殘差塊（兩層 3×3 卷積）。

    結構：
        Conv3×3(stride) → BN → ReLU → Conv3×3 → BN → (+shortcut) → ReLU

    當 stride != 1 或 channel 數改變時，shortcut 用 1×1 Conv + BN 做投影對齊。
    """

    expansion = 1  # ResNet-34 的 BasicBlock 不做 channel 擴展

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        # 第一層卷積：可能有 stride（用於下採樣）
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        # 第二層卷積：stride=1，維持尺寸
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        # Shortcut projection：當 stride != 1 或 channel 數改變時需要對齊
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        # 主路徑：兩層卷積 + BN + ReLU
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # 殘差連接：主路徑 + shortcut，再 ReLU
        out += self.shortcut(x)
        return self.relu(out)


def _make_layer(in_ch, out_ch, num_blocks, stride=1):
    """建立 ResNet 的一個 stage（多個 BasicBlock 串聯）。

    只有第一個 block 可能有 stride（其餘 stride=1）。
    """
    layers = [BasicBlock(in_ch, out_ch, stride)]       # 第一個 block，可能含 stride
    for _ in range(1, num_blocks):
        layers.append(BasicBlock(out_ch, out_ch, stride=1))  # 後續 block，stride=1
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# UNet 風格的 Decoder Block
# ---------------------------------------------------------------------------

class DecoderBlock(nn.Module):
    """上採樣 → concat skip connection → Conv-BN-ReLU × 2。

    流程：
        1. Bilinear upsample（scale_factor=2）
        2. 若與 skip connection 尺寸不符，用 F.pad 補齊
        3. concat skip connection（channel 維度）
        4. 兩層 Conv3×3-BN-ReLU 做特徵融合
    """

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        # Bilinear 上採樣，空間尺寸 ×2
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        # 兩層 Conv3×3（有 padding=1 維持尺寸），含 BN 和 ReLU
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip=None):
        # 上採樣
        x = self.up(x)
        if skip is not None:
            # 若尺寸有差（奇數尺寸的 edge case），在右邊和下面補 padding 對齊
            if x.shape[2:] != skip.shape[2:]:
                x = F.pad(x, [0, skip.shape[3] - x.shape[3],
                              0, skip.shape[2] - x.shape[2]])
            # channel 維度 concat
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ---------------------------------------------------------------------------
# ResNet34 + UNet 完整模型
# ---------------------------------------------------------------------------

class ResNet34UNet(nn.Module):
    """ResNet-34 Encoder 搭配 UNet 風格 Decoder。

    以 256×256 輸入為例的特徵圖尺寸：
        Encoder（ResNet-34）：
            stem  : 64ch,  128×128  （7×7 Conv, stride=2 + BN + ReLU）
            pool  : 64ch,   64×64   （MaxPool 3×3, stride=2）
            layer1: 64ch,   64×64   （3 × BasicBlock, stride=1）
            layer2: 128ch,  32×32   （4 × BasicBlock, stride=2）
            layer3: 256ch,  16×16   （6 × BasicBlock, stride=2）
            layer4: 512ch,   8×8    （3 × BasicBlock, stride=2）

        Decoder（UNet 風格）：
            dec4: 512 → 256   （skip: layer3, 256ch, 16×16）
            dec3: 256 → 128   （skip: layer2, 128ch, 32×32）
            dec2: 128 →  64   （skip: layer1,  64ch, 64×64）
            dec1:  64 →  64   （skip: stem,    64ch, 128×128）
            dec0:  64 →  32   （無 skip，輸出 256×256）
            out:   32 →   1   （Conv1×1 + sigmoid）
    """

    # ResNet-34 各 stage 的 BasicBlock 數量
    LAYERS = [3, 4, 6, 3]

    def __init__(self, in_channels=3):
        super().__init__()

        # ---- Encoder：ResNet-34 ----

        # Stem：7×7 Conv, stride=2 → 空間縮小一半（256→128）
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        # MaxPool stride=2 → 空間再縮小一半（128→64）
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 四個 ResNet stage
        self.layer1 = _make_layer(64,  64,  self.LAYERS[0], stride=1)  # 64×64,  64ch
        self.layer2 = _make_layer(64,  128, self.LAYERS[1], stride=2)  # 32×32, 128ch
        self.layer3 = _make_layer(128, 256, self.LAYERS[2], stride=2)  # 16×16, 256ch
        self.layer4 = _make_layer(256, 512, self.LAYERS[3], stride=2)  #  8×8,  512ch

        # ---- Decoder：UNet 風格 ----
        # 每個 DecoderBlock(in_ch, skip_ch, out_ch)
        self.dec4 = DecoderBlock(512, 256, 256)   # 8×8   → 16×16
        self.dec3 = DecoderBlock(256, 128, 128)   # 16×16 → 32×32
        self.dec2 = DecoderBlock(128,  64,  64)   # 32×32 → 64×64
        self.dec1 = DecoderBlock( 64,  64,  64)   # 64×64 → 128×128，skip 接 stem
        self.dec0 = DecoderBlock( 64,   0,  32)   # 128×128 → 256×256，無 skip

        # 1×1 輸出卷積，32ch → 1ch
        self.out_conv = nn.Conv2d(32, 1, kernel_size=1)

        # Kaiming 初始化所有卷積層
        self._init_weights()

    def _init_weights(self):
        """Kaiming Normal 初始化所有 Conv2d 層，ones/zeros 初始化 BN。"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                # BN 的 weight（gamma）初始化為 1，bias（beta）初始化為 0
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # ---- Encoder 前向傳遞 ----
        s = self.stem(x)          # (B, 64,  H/2,  W/2)  — stem 特徵，供 dec1 skip
        p = self.pool(s)          # (B, 64,  H/4,  W/4)

        e1 = self.layer1(p)       # (B, 64,  H/4,  W/4)  — layer1 特徵，供 dec2 skip
        e2 = self.layer2(e1)      # (B, 128, H/8,  W/8)  — layer2 特徵，供 dec3 skip
        e3 = self.layer3(e2)      # (B, 256, H/16, W/16) — layer3 特徵，供 dec4 skip
        e4 = self.layer4(e3)      # (B, 512, H/32, W/32) — 最深層特徵

        # ---- Decoder 前向傳遞 ----
        d4 = self.dec4(e4, e3)    # (B, 256, H/16, W/16)
        d3 = self.dec3(d4, e2)    # (B, 128, H/8,  W/8)
        d2 = self.dec2(d3, e1)    # (B, 64,  H/4,  W/4)
        d1 = self.dec1(d2, s)     # (B, 64,  H/2,  W/2)  — skip 接 stem
        d0 = self.dec0(d1)        # (B, 32,  H,    W)    — 無 skip，回到原始尺寸

        # 1×1 卷積 + Sigmoid → 輸出 [0,1] 機率圖
        return torch.sigmoid(self.out_conv(d0))


if __name__ == "__main__":
    # 快速測試：確認輸入輸出 shape 與參數量
    model = ResNet34UNet()
    x = torch.randn(2, 3, 256, 256)
    y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}")
