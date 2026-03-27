"""在驗證集上評估已訓練模型的 Dice Score。

使用方式：
    python src/evaluate.py --model unet \\
        --model_path saved_models/unet_best.pth \\
        --list_dir dataset/oxford-iiit-pet/nycu-2026-spring-dl-lab2-unet --gpu 0
    python src/evaluate.py --model resnet34_unet \\
        --model_path saved_models/resnet34_unet_best.pth \\
        --list_dir dataset/oxford-iiit-pet/nycu-2026-spring-dl-lab2-resnet34unet --gpu 1
"""

import os
import sys
import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# 讓 Python 能找到同目錄下的其他模組
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from oxford_pet import OxfordPetDataset
from utils import dice_score, load_checkpoint
from models.unet import UNet
from models.resnet34_unet import ResNet34UNet


def get_model(name):
    """根據名稱回傳對應的模型實例。"""
    if name == "unet":
        return UNet()
    elif name == "resnet34_unet":
        return ResNet34UNet()
    else:
        raise ValueError(f"未知的模型: {name}")


@torch.no_grad()
def evaluate(args):
    """載入 checkpoint 並在驗證集上計算 Dice Score。

    @torch.no_grad() 裝飾器關閉梯度計算，節省記憶體。
    """

    # ---- 選擇裝置 ----
    if args.gpu is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"裝置: {device}")

    # ---- 建立驗證集 DataLoader ----
    # list_dir 指定課程提供的 split 檔案目錄
    val_ds = OxfordPetDataset(args.data_path, mode="val",
                              img_size=args.img_size, list_dir=args.list_dir)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    print(f"驗證集圖片數量: {len(val_ds)}")

    # ---- 載入模型和 checkpoint ----
    model = get_model(args.model).to(device)
    ckpt  = load_checkpoint(model, args.model_path, device)
    # 從 checkpoint 讀取訓練時紀錄的 epoch 和 val Dice（供比對用）
    saved_epoch = ckpt.get("epoch", "?")
    saved_dice  = ckpt.get("val_dice", float("nan"))
    print(f"載入 checkpoint: epoch={saved_epoch}, 儲存時 Val Dice={saved_dice:.4f}")
    model.eval()

    # ---- 逐 batch 計算 Dice ----
    all_dice = []
    for images, masks in tqdm(val_loader, desc="Evaluating"):
        images = images.to(device)
        masks  = masks.to(device)
        preds  = model(images)
        # 每個 batch 的平均 Dice 存入清單
        all_dice.append(dice_score(preds, masks))

    # 注意：這裡是對 batch-level Dice 取平均（而非 global Dice）
    # 嚴格來說，batch size 不等時兩者結果略有差異
    mean_dice = sum(all_dice) / len(all_dice)
    print(f"\n驗證集平均 Dice Score: {mean_dice:.4f}")
    return mean_dice


def parse_args():
    """解析命令列參數。"""
    parser = argparse.ArgumentParser(description="在驗證集上評估分割模型")

    parser.add_argument(
        "--model", type=str, default="unet",
        choices=["unet", "resnet34_unet"],
        help="模型架構"
    )
    parser.add_argument(
        "--model_path", type=str, required=True,
        help=".pth checkpoint 路徑"
    )
    parser.add_argument(
        "--data_path", type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "dataset", "oxford-iiit-pet"
        ),
        help="Oxford-IIIT Pet 資料集根目錄"
    )
    parser.add_argument("--batch_size",  type=int, default=16,  help="Batch size")
    parser.add_argument("--img_size",    type=int, default=256, help="圖片縮放大小")
    parser.add_argument("--num_workers", type=int, default=4,   help="DataLoader worker 數")
    parser.add_argument(
        "--gpu", type=int, default=None,
        help="指定 GPU 編號（如 0 或 1），不指定時自動使用 cuda:0"
    )
    parser.add_argument(
        "--list_dir", type=str, default=None,
        help=(
            "課程提供的 split 檔案目錄，包含 train.txt / val.txt / test_*.txt。"
            "例如：dataset/oxford-iiit-pet/nycu-2026-spring-dl-lab2-unet"
        )
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
