# Lab2 Binary Semantic Segmentation — 實作總覽

> 最後更新：2026-03-24

---

## 專案結構

```
Lab2_BinarySemanticSegmentation/
├── dataset/
│   └── oxford-iiit-pet/
│       ├── images/               # .jpg 原始圖片
│       └── annotations/
│           ├── trimaps/          # .png trimap 標註
│           ├── trainval.txt      # train+val 清單
│           └── test.txt          # test 清單
├── src/
│   ├── oxford_pet.py             # Dataset 定義
│   ├── utils.py                  # Dice / Loss / RLE / Checkpoint
│   ├── train.py                  # 訓練腳本
│   ├── evaluate.py               # 驗證集評估
│   ├── inference.py              # 測試集推論 → Kaggle CSV
│   └── models/
│       ├── unet.py               # 嚴格 2015 UNet
│       └── resnet34_unet.py      # ResNet34 Encoder + UNet Decoder
├── saved_models/                 # 訓練產出的 .pth checkpoint
└── predictions/                  # inference.py 產出的 CSV
```

---

## 資料集：Oxford-IIIT Pet

| 項目 | 說明 |
|------|------|
| 圖片格式 | RGB JPEG |
| 標註格式 | Trimap PNG（pixel 值 1/2/3） |
| Binary 轉換 | pixel==1 → 前景(1.0)，其他 → 背景(0.0) |
| Train/Val 切分 | 80/20，從 trainval.txt，`seed=42`，可重現 |
| Test 集 | test.txt，無 mask，只回傳 (image, filename) |
| 圖片大小 | resize 至 256×256 |
| 正規化 | ImageNet mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225] |

### 資料增強（僅 Train）

| 增強方法 | 參數 | 套用機率 |
|----------|------|---------|
| 水平翻轉 | — | 50% |
| 垂直翻轉 | — | 50% |
| 隨機旋轉 | ±15° | 50% |
| 隨機裁切 resize | scale=(0.7,1.0), ratio=(0.75,1.33) | 50% |
| Color Jitter | brightness/contrast/saturation=0.3, hue=0.05 | 100% |

> **注意**：旋轉和裁切必須對 image 和 mask 套用相同的幾何變換；Color Jitter 只套用於 image，mask 不動。

---

## Model 1：UNet（嚴格遵照 2015 論文）

### 架構規則（Kaggle 討論區要求）

| 規則 | 說明 |
|------|------|
| **禁止** BatchNorm | 原始論文沒有 |
| **禁止** padding | 所有 Conv2d 使用 padding=0 |
| **必須** ConvTranspose2d | 上採樣用 2×2 transposed conv，不可用 bilinear |
| **必須** copy-and-crop | skip connection 用 center crop，不是 padding |
| 允許 in_channels=3 | RGB 輸入（原論文是灰階） |
| 允許 out_channels=1 | Binary sigmoid（原論文是 2 channel softmax） |

### 架構細節（256×256 輸入）

```
Encoder:
  enc1: 3   → 64  ch  (ConvBlock: Conv3×3+ReLU × 2, padding=0)
  enc2: 64  → 128 ch  (MaxPool2d(2) + ConvBlock)
  enc3: 128 → 256 ch  (MaxPool2d(2) + ConvBlock)
  enc4: 256 → 512 ch  (MaxPool2d(2) + ConvBlock)
Bottleneck:
  512 → 1024 ch       (MaxPool2d(2) + ConvBlock)
Decoder:
  dec4: 1024 → 512 ch (ConvTranspose2d(2×2) + center_crop + cat + ConvBlock)
  dec3:  512 → 256 ch (同上)
  dec2:  256 → 128 ch (同上)
  dec1:  128 →  64 ch (同上)
Output:
  64 → 1 ch (Conv1×1)
  F.interpolate 回原始 input size（後處理，不算網路架構）
  torch.sigmoid → 輸出 [0,1] 機率
```

**參數量**：31,031,745

### 關鍵注意事項

- `ConvBlock`：兩個 3×3 Conv + ReLU，**無 padding，無 BN**，每個 block 空間縮小 4px
- `_center_crop`：encoder feature map 比 decoder 大（因 unpadded conv），需 center crop 後才能 concat
- `forward()` 末尾的 `F.interpolate`：不是模型結構的一部分，是為了對齊 loss 計算而做的後處理（因 unpadded conv 導致輸出比輸入小）
- 最終輸出 `torch.sigmoid(out)` → 機率值在 [0, 1]

---

## Model 2：ResNet34 + UNet Decoder

### 架構細節（256×256 輸入）

```
Encoder (ResNet-34):
  stem:   Conv7×7(stride=2) + BN + ReLU  → 64ch,  128×128
  pool:   MaxPool3×3(stride=2)            → 64ch,   64×64
  layer1: [3 × BasicBlock, stride=1]      → 64ch,   64×64
  layer2: [4 × BasicBlock, stride=2]      → 128ch,  32×32
  layer3: [6 × BasicBlock, stride=2]      → 256ch,  16×16
  layer4: [3 × BasicBlock, stride=2]      → 512ch,   8×8

Decoder (UNet-style DecoderBlock):
  dec4: 512 → 256  (skip: layer3, 256ch,  16×16)
  dec3: 256 → 128  (skip: layer2, 128ch,  32×32)
  dec2: 128 →  64  (skip: layer1,  64ch,  64×64)
  dec1:  64 →  64  (skip: stem,    64ch, 128×128)
  dec0:  64 →  32  (no skip,             256×256)
  out:   32 →   1  (Conv1×1 + sigmoid)
```

**參數量**：24,521,697

### BasicBlock（ResNet 殘差塊）

```
Conv3×3(stride) → BN → ReLU → Conv3×3 → BN → (+shortcut) → ReLU
shortcut: Identity，或 Conv1×1+BN（channel 數改變 or stride≠1 時）
```

### DecoderBlock

```
Upsample(scale=2, bilinear) → F.pad（防 size mismatch）→ cat(skip) → Conv3×3-BN-ReLU × 2
```

### 與 UNet 的差異

| | UNet | ResNet34+UNet |
|---|---|---|
| BatchNorm | 無 | 有 |
| Padding | 無 (padding=0) | 有 (padding=1) |
| 上採樣 | ConvTranspose2d | Upsample(bilinear) |
| Skip Connection | center-crop | F.pad（保護 size mismatch） |
| 權重初始化 | PyTorch 預設 | Kaiming Normal |
| dec0 skip | 無此層 | 無 skip（skip_ch=0） |
| 收斂速度 | 極慢，不穩定 | 快速穩定 |

---

## 工具函式（utils.py）

| 函式 | 說明 |
|------|------|
| `dice_score(pred, target, threshold=0.5)` | 先 threshold 二值化，計算 per-image Dice 後取 batch 平均，回傳 float |
| `dice_loss(pred, target)` | 使用機率值（不二值化）計算可微分 Dice Loss = 1 − Dice |
| `combined_loss(pred, target, bce_weight=0.5)` | 0.5 × BCE + 0.5 × DiceLoss |
| `rle_encode(mask)` | Column-major RLE，Kaggle submission 格式 |
| `save_checkpoint(...)` | 儲存 epoch, model weights, optimizer state, val_dice |
| `load_checkpoint(model, path, device)` | 載入 model weights，回傳完整 checkpoint dict |

> `smooth=1e-6`：防止分母為零（全黑 mask 的邊緣情況）

---

## 訓練、驗證、推論流程

```
Train（train.py）  →  Checkpoint（saved_models/*.pth）
                           ↓
                    Evaluate（evaluate.py）  →  Val Dice 分數
                           ↓（選最佳 checkpoint）
                    Inference（inference.py）  →  predictions/submission.csv
                           ↓
                    上傳 Kaggle 評分
```

### 常用指令

```bash
# 訓練
python src/train.py --model unet --gpu 0 --epochs 100 --batch_size 32 --lr 1e-4
python src/train.py --model resnet34_unet --gpu 1 --epochs 50 --batch_size 16 --lr 5e-4

# 驗證
python src/evaluate.py --model unet --model_path saved_models/unet_MMDD_HHMM_best.pth
python src/evaluate.py --model resnet34_unet --model_path saved_models/resnet34_unet_MMDD_HHMM_best.pth

# 推論（產生 Kaggle CSV）
python src/inference.py --model unet --model_path saved_models/unet_MMDD_HHMM_best.pth
python src/inference.py --model resnet34_unet --model_path saved_models/resnet34_unet_MMDD_HHMM_best.pth
```

**訓練固定設定**：
- Optimizer: Adam, weight_decay=1e-5
- Scheduler: CosineAnnealingLR, T_max=epochs, eta_min=1e-6
- Loss: combined_loss（BCE 0.5 + Dice 0.5）
- Checkpoint 命名：`{model}_{MMDD_HHMM}_best.pth`（只儲存 val Dice 最佳 epoch）

---

## 實驗紀錄

### UNet 實驗

| | Exp 1 | Exp 2 | Exp 3（計畫） |
|---|---|---|---|
| 日期 | 2026-03-24 | 2026-03-24 | — |
| Checkpoint | unet_0324_1135_best.pth | unet_0324_XXXX_best.pth | — |
| epochs | 50 | 100 | 150 |
| batch_size | 32 | 32 | 32 |
| **lr** | **1e-3** | **1e-4** | **5e-5** |
| Gradient Clipping | 無 | 無 | **有（max_norm=1.0）** |
| Best Val Dice | 0.2111（Ep.45） | 0.2606（Ep.77） | — |
| 最終 Train Dice | 0.2374 | 0.2766 | — |
| 最終 Train Loss | 0.6366 | 0.6345 | — |
| **結果** | **失敗** | **仍然很差** | 等待中 |

**UNet 失敗根本原因**：
1. 無 BatchNorm → 各層 activation 尺度差異大 → 梯度噪音高
2. 缺乏 Gradient Clipping → 梯度爆炸 → Dice 劇烈跳動
3. Loss 幾乎不動（~0.64）：模型陷入不良局部最小值，幾乎沒有有效學習

### ResNet34+UNet 實驗

| | Exp 1 |
|---|---|
| 日期 | 2026-03-24 |
| epochs | 50 |
| batch_size | 16 |
| lr | 5e-4 |
| Best Val Dice | **0.8904**（Ep.49） |
| 最終 Train Dice | 0.9069 |
| 最終 Train Loss | 0.1253 |
| **結果** | **成功** |

**ResNet34 成功原因**：有 BN 穩定梯度、Kaiming init、Residual connection，第 1 個 epoch 即達 Val Dice 0.69。

---

## 環境

| 項目 | 說明 |
|------|------|
| GPU | GTX 1080 Ti × 2（各 11GB） |
| GPU 分配 | UNet → GPU 0，ResNet34 → GPU 1 |
| Python | Anaconda base env（內含 torch + torchvision） |
| 注意 | VSCode terminal 可能使用不同 Python，執行前先確認 conda env |
| GPU 監控 | `nvidia-smi -l 1` 或 `nvitop` |

---

## 待辦事項

- [ ] UNet Exp3：加入 gradient clipping，lr=5e-5，epochs=150
- [ ] ResNet34 Exp1 evaluate：`python src/evaluate.py --model resnet34_unet ...`
- [ ] ResNet34 Exp1 inference：`python src/inference.py --model resnet34_unet ...`
- [ ] 上傳 ResNet34 結果至 Kaggle
- [ ] 待 UNet Exp3 完成後再上傳 UNet 結果
