import torch
import numpy as np


def dice_score(pred, target, threshold=0.5, smooth=1e-6):
    # 計算 Dice Similarity Coefficient。

    # 公式：Dice = 2 * |X ∩ Y| / (|X| + |Y|)

    # 先用 threshold 將預測機率binary化，再與 ground truth 比較。
    # 適合用於 validation / evaluation，不用於 loss 計算（因為不可微分）。

    # Args:
    #     pred      (Tensor): 模型輸出的機率值，shape (B, 1, H, W)，值域 [0, 1]。
    #     target    (Tensor): Ground truth binary mask，shape (B, 1, H, W)，值 0 或 1。
    #     threshold (float):  二值化閾值，預設 0.5。
    #     smooth    (float):  平滑項，防止分母為 0（全黑 mask 的邊緣情況）。

    # Returns:
    #     float: batch 內各圖片 Dice 分數的平均值。
    
    # 用 threshold 把機率轉成 0 或 1（不可微分操作）
    pred_bin = (pred > threshold).float()
    target   = target.float()

    # 在空間維度（H, W）上計算交集和聯集，dim=(1,2,3) 對 B 以外的維度求和
    intersection = (pred_bin * target).sum(dim=(1, 2, 3))
    union        = pred_bin.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))

    # 加 smooth 避免全黑 mask 時的除零問題
    dice = (2.0 * intersection + smooth) / (union + smooth)
    # 對 batch 取平均，回傳 Python float
    return dice.mean().item()


def dice_loss(pred, target, smooth=1e-6):
    # 可微分的 Dice Loss（使用機率值，不二值化）。

    # 與 dice_score 不同，這裡直接使用 sigmoid 輸出的連續機率值做計算，
    # 使損失函式對整個網路可微分，可以反向傳播。

    # 公式：DiceLoss = 1 - Dice

    # Args:
    #     pred   (Tensor): Sigmoid 機率值，shape (B, 1, H, W)。
    #     target (Tensor): Ground truth binary mask，shape (B, 1, H, W)。
    #     smooth (float):  平滑項，防止除零。

    # Returns:
    #     Tensor: 純量 Dice Loss，值域 [0, 1]，值越小越好。
    
    pred   = pred.float()
    target = target.float()

    # 用連續機率值計算「軟」交集
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union        = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))

    dice = (2.0 * intersection + smooth) / (union + smooth)
    # 1 - Dice 作為 loss（越小越好），對 batch 取平均
    return (1.0 - dice).mean()

#Claude Co-work
def combined_loss(pred, target, bce_weight=0.5):
    # BCE Loss 和 Dice Loss 的加權combine loss。

    # 公式：combined = bce_weight × WeightedBCE + (1 - bce_weight) × DiceLoss

    # 結合兩種 loss 的優點：
    # - WeightedBCE：自動計算 pos_weight 補償前景/背景不平衡，防止 mode collapse
    # - Dice：直接優化評估指標，對正負樣本不平衡有魯棒性

    # 【關鍵說明】pos_weight 的必要性：
    #     Oxford Pet 資料集中背景像素約佔 70-80%，前景僅 20-30%。
    #     若不加 pos_weight，BCE 梯度被大量背景像素主導，
    #     模型會學到「全部預測背景」是降低 loss 的捷徑（mode collapse）。
    #     動態計算 pos_weight = 背景數 / 前景數 可讓兩類梯度貢獻相等，
    #     迫使模型真正學習辨識前景。

    # Args:
    #     pred       (Tensor): Sigmoid 機率值，shape (B, 1, H, W)。
    #     target     (Tensor): Ground truth binary mask，shape (B, 1, H, W)。
    #     bce_weight (float):  BCE loss 的權重，Dice loss 權重 = 1 - bce_weight。
    #                          預設 0.5 代表各佔一半。

    # Returns:
    #     Tensor: 純量組合損失。
    
    import torch
    import torch.nn.functional as F

    # 動態計算 pos_weight（前景補償係數）
    # 前景像素少 → pos_weight 大 → 前景梯度被放大 → 防止 mode collapse
    num_pos = target.sum() + 1e-6          # batch 內前景像素總數
    num_neg = (1.0 - target).sum() + 1e-6  # batch 內背景像素總數
    # clamp(max=10) 防止極端不平衡時梯度爆炸
    pos_weight = (num_neg / num_pos).clamp(max=10.0)

    # 建立 per-pixel weight map：前景像素 × pos_weight，背景像素 × 1
    weight = torch.where(target > 0.5,
                         pos_weight * torch.ones_like(target),
                         torch.ones_like(target))
    # Weighted BCE（使用 F.binary_cross_entropy 支援 per-pixel weight）
    bce    = F.binary_cross_entropy(pred, target, weight=weight)
    d_loss = dice_loss(pred, target)       # Soft Dice Loss
    return bce_weight * bce + (1.0 - bce_weight) * d_loss

# Claude Co-work
def rle_encode(mask):
    # 將 binary mask 做 Run-Length Encoding，用於 Kaggle 提交。

    # 採用 column-major順序( Kaggle 分割競賽的標準格式)，編碼格式為「起始位置 長度 起始位置 長度 ...」（1-indexed）。

    # Args:
    #     mask (np.ndarray): 2D binary mask，shape (H, W)，值 0 或 1。

    # Returns:
    #     str: RLE 編碼字串，例如 "5539 6 5794 18 ..."。
    #          若 mask 全為 0（無前景），回傳空字串。
    
    # 按 Fortran（column-major）順序攤平：先遍歷欄，再遍歷列
    pixels = mask.flatten(order="F")
    # 在頭尾各加一個 0，方便偵測邊界
    pixels = np.concatenate([[0], pixels, [0]])
    # 找到所有 0→1 或 1→0 的轉換位置（1-indexed）
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    # 奇數索引位置（長度）= 結束位置 - 起始位置
    runs[1::2] -= runs[::2]
    return " ".join(str(r) for r in runs)


def save_checkpoint(model, optimizer, epoch, val_dice, path):
    # 儲存模型 checkpoint（weights + optimizer state + 訓練資訊）。

    torch.save(
        {
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_dice":             val_dice,
        },
        path,
    )


def load_checkpoint(model, path, device):
    # map_location 可以確保在不同裝置間載入（例如在 CPU 上載入 GPU 訓練的 checkpoint）
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    return ckpt
