#使用方式：
#     python src/train.py --model unet --epochs 50 --batch_size 16 --lr 1e-3 --list_dir dataset/oxford-iiit-pet/nycu-2026-spring-dl-lab2-unet
#     python src/train.py --model resnet34_unet --epochs 50 --batch_size 8 --lr 5e-4 --list_dir dataset/oxford-iiit-pet/nycu-2026-spring-dl-lab2-resnet34unet

# 固定訓練設定：
#     - Optimizer:  Adam, weight_decay=1e-5
#     - Scheduler:  CosineAnnealingLR, T_max=epochs, eta_min=1e-6
#     - Loss:       combined_loss(BCE 0.5 + Dice 0.5)
#     - Checkpoint: 只在 val Dice 更新最佳值時儲存


import os
import sys
import argparse
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from oxford_pet import OxfordPetDataset
from utils import dice_score, combined_loss, save_checkpoint
from models.unet import UNet
from models.resnet34_unet import ResNet34UNet


def get_model(name):
    if name == "unet":
        return UNet()
    elif name == "resnet34_unet":
        return ResNet34UNet()
    else:
        raise ValueError(f"Unknown model: {name}")


def train_one_epoch(model, loader, optimizer, device, bce_weight=0.5):
    # 執行一個 epoch 的訓練，回傳平均 loss 和平均 Dice score。
    model.train()
    total_loss = 0.0
    total_dice = 0.0

    for images, masks in tqdm(loader, desc="Train", leave=False):
        # 將資料移到指定裝置（GPU/CPU）
        images = images.to(device)
        masks  = masks.to(device)

        optimizer.zero_grad()
        preds = model(images)                               # 前向傳遞
        loss  = combined_loss(preds, masks, bce_weight=bce_weight)  # 計算 loss
        loss.backward()                                     # 反向傳播
        # Gradient clipping：防止無 BN 的深層網路梯度爆炸，穩定訓練
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()                                    # 更新權重

        total_loss += loss.item()
        # detach() 確保計算 Dice 時不建立計算圖（節省記憶體）
        total_dice += dice_score(preds.detach(), masks.detach())

    n = len(loader)  # batch 數量
    return total_loss / n, total_dice / n


@torch.no_grad()
def validate(model, loader, device, bce_weight=0.5):
    # 在驗證集上評估模型，回傳平均 loss 和平均 Dice score。
    # Args:
    #     bce_weight (float): BCE 在 combined_loss 中的權重（預設 0.5）。
   
    model.eval()  # 切換為評估模式（關閉 Dropout / BatchNorm 的訓練行為）
    total_loss = 0.0
    total_dice = 0.0

    for images, masks in tqdm(loader, desc="Val  ", leave=False):
        images = images.to(device)
        masks  = masks.to(device)

        preds = model(images)
        loss  = combined_loss(preds, masks, bce_weight=bce_weight)

        total_loss += loss.item()
        total_dice += dice_score(preds, masks)

    n = len(loader)
    return total_loss / n, total_dice / n


def train(args):
    """完整訓練流程。"""

    # ---- 選擇裝置 ----
    if args.gpu is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- 建立 Dataset 和 DataLoader ----
    from torch.utils.data import DataLoader
    train_ds = OxfordPetDataset(args.data_path, mode="train",
                                img_size=args.img_size, list_dir=args.list_dir)
    val_ds   = OxfordPetDataset(args.data_path, mode="val",
                                img_size=args.img_size, list_dir=args.list_dir)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True   # pin_memory 加速 GPU 傳輸
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    print(f"Train: {len(train_ds)} images | Val: {len(val_ds)} images")

    # ---- 建立模型 ----
    model = get_model(args.model).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model} | Parameters: {total_params:,}")

    # ---- 優化器 & 學習率排程器 ----
    # Adam + weight decay 防止過擬合
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    # CosineAnnealingLR：學習率從 lr 餘弦退火到 eta_min=1e-6
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ---- 設定 checkpoint 儲存路徑 ----
    os.makedirs(args.save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%m%d_%H%M")
    best_path = os.path.join(args.save_dir, f"{args.model}_{timestamp}_best.pth")

    best_dice = 0.0  # 紀錄最佳 val Dice，用於判斷是否要儲存 checkpoint

    # ---- 訓練迴圈 ----
    for epoch in range(1, args.epochs + 1):
        # 傳入 bce_weight：控制 BCE vs Dice Loss 的比例
        train_loss, train_dice = train_one_epoch(
            model, train_loader, optimizer, device, bce_weight=args.bce_weight)
        val_loss, val_dice = validate(
            model, val_loader, device, bce_weight=args.bce_weight)
        scheduler.step()  # 更新學習率

        print(
            f"Epoch [{epoch:>3}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f}  Train Dice: {train_dice:.4f}  "
            f"Val Loss: {val_loss:.4f}  Val Dice: {val_dice:.4f}"
        )

        if val_dice > best_dice:
            best_dice = val_dice
            save_checkpoint(model, optimizer, epoch, val_dice, best_path)
            print(f"  -> Saved best model (Val Dice: {best_dice:.4f})")

    print(f"\nTraining done. Best Val Dice: {best_dice:.4f}")
    print(f"Best model saved to: {best_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train segmentation model")

    parser.add_argument(
        "--model", type=str, default="unet",
        choices=["unet", "resnet34_unet"],
        help="Model Architecture"
    )
    parser.add_argument(
        "--data_path", type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "dataset", "oxford-iiit-pet"
        ),
        help="Oxford-IIIT Pet 資料集根目錄"
    )
    parser.add_argument(
        "--save_dir", type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "saved_models"
        ),
        help="模型 checkpoint 儲存目錄"
    )
    parser.add_argument("--epochs",      type=int,   default=50,   help="訓練 epoch 數")
    parser.add_argument("--batch_size",  type=int,   default=16,   help="Batch size")
    parser.add_argument("--lr",          type=float, default=1e-3, help="初始學習率")
    parser.add_argument(
        "--bce_weight", type=float, default=0.5,
        help=(
            "combined_loss 中 BCE 的權重（0~1）。"
            "Dice weight = 1 - bce_weight。"
            "推薦 UNet（無 BN）使用 0.3；設 0 則為純 Dice Loss。"
            "預設 0.5。"
        )
    )
    parser.add_argument("--img_size",    type=int,   default=256,  help="圖片縮放大小")
    parser.add_argument("--num_workers", type=int,   default=4,    help="DataLoader worker 數")
    parser.add_argument(
        "--gpu", type=int, default=None,
        help="指定 GPU 編號（如 0 或 1），不指定時自動使用 cuda:0"
    )
    parser.add_argument(
        "--list_dir", type=str, default=None,
        help=(
            "課程提供的 split 檔案目錄，包含 train.txt / val.txt / test_*.txt。"
            "例如：dataset/oxford-iiit-pet/nycu-2026-spring-dl-lab2-unet  "
            "不指定時使用原始 trainval.txt 的 80/20 切分。"
        )
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
