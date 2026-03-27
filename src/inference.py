"""對測試集進行推論並輸出 Kaggle 格式的 CSV 檔。

使用方式：
    python src/inference.py --model unet \\
        --model_path saved_models/unet_best.pth \\
        --list_dir dataset/oxford-iiit-pet/nycu-2026-spring-dl-lab2-unet --gpu 0
    python src/inference.py --model resnet34_unet \\
        --model_path saved_models/resnet34_unet_best.pth \\
        --list_dir dataset/oxford-iiit-pet/nycu-2026-spring-dl-lab2-resnet34unet --gpu 1

輸出：
    predictions/<model_name>_submission.csv

CSV 格式（Kaggle 要求）：
    image_id,encoded_mask
    Bengal_61,5539 6 5794 18 ...
    ...

【重要】mask resize 回原始尺寸：
    模型推論時輸入為 256×256，但 Oxford Pet 原始圖片各自有不同尺寸。
    Kaggle 用原始尺寸解碼 RLE，因此提交前必須將 mask resize 回原始大小。
"""

import os
import sys
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image  # 用於 resize 預測 mask

# 讓 Python 能找到同目錄下的其他模組
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from oxford_pet import OxfordPetDataset
from utils import rle_encode, load_checkpoint
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
def inference(args):
    """載入 checkpoint，對測試集推論並輸出 Kaggle 提交 CSV。"""

    # ---- 選擇裝置 ----
    if args.gpu is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"裝置: {device}")

    # ---- 建立測試集 DataLoader ----
    # test mode 不需要 mask，回傳 (image, filename, orig_w, orig_h)
    test_ds = OxfordPetDataset(args.data_path, mode="test",
                               img_size=args.img_size, list_dir=args.list_dir)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    print(f"測試集圖片數量: {len(test_ds)}")

    # ---- 載入模型和 checkpoint ----
    model = get_model(args.model).to(device)
    load_checkpoint(model, args.model_path, device)
    model.eval()
    print(f"模型 {args.model} 載入完成: {args.model_path}")

    # ---- 建立輸出資料夾 ----
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, f"{args.model}_submission.csv")

    rows = []  # 儲存 (image_id, rle_string) 的清單

    # ---- 逐 batch 推論 ----
    for images, names, orig_ws, orig_hs in tqdm(test_loader, desc="Inference"):
        images = images.to(device)
        preds  = model(images)   # shape: (B, 1, 256, 256)，機率值 [0,1]

        # 用 threshold=0.5 二值化，並轉為 numpy uint8
        preds_bin = (preds > args.threshold).squeeze(1).cpu().numpy().astype(np.uint8)

        for pred_mask, name, orig_w, orig_h in zip(preds_bin, names, orig_ws, orig_hs):
            orig_w = int(orig_w)
            orig_h = int(orig_h)

            # 【關鍵步驟】將 256×256 的預測 mask resize 回原始圖片尺寸
            # Kaggle 用原始圖片尺寸解碼 RLE，若不 resize 則 mask 完全錯位
            if pred_mask.shape != (orig_h, orig_w):
                pil_mask = Image.fromarray(pred_mask)
                # NEAREST 插值保持 binary 值（0 或 1）不被插值成中間值
                pil_mask  = pil_mask.resize((orig_w, orig_h), Image.NEAREST)
                pred_mask = np.array(pil_mask, dtype=np.uint8)

            # RLE 編碼（column-major，Kaggle 標準格式）
            rle = rle_encode(pred_mask)
            rows.append((name, rle))

    # ---- 寫入 CSV ----
    with open(csv_path, "w") as f:
        # Kaggle 要求的欄位名稱：image_id 和 encoded_mask
        f.write("image_id,encoded_mask\n")
        for image_id, rle in rows:
            f.write(f"{image_id},{rle}\n")

    print(f"\n推論完成，共 {len(rows)} 張圖片。")
    print(f"Kaggle 提交檔案儲存至: {csv_path}")


def parse_args():
    """解析命令列參數。"""
    parser = argparse.ArgumentParser(description="對測試集進行推論並輸出 Kaggle CSV")

    parser.add_argument(
        "--model", type=str, required=True,
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
    parser.add_argument(
        "--output_dir", type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "predictions"
        ),
        help="輸出 CSV 的資料夾"
    )
    parser.add_argument("--threshold",   type=float, default=0.5,
                        help="二值化閾值（預設 0.5）")
    parser.add_argument("--batch_size",  type=int,   default=16,  help="Batch size")
    parser.add_argument("--img_size",    type=int,   default=256, help="模型輸入圖片大小")
    parser.add_argument("--num_workers", type=int,   default=4,   help="DataLoader worker 數")
    parser.add_argument(
        "--gpu", type=int, default=None,
        help="指定 GPU 編號（如 0 或 1），不指定時自動使用 cuda:0"
    )
    parser.add_argument(
        "--list_dir", type=str, default=None,
        help=(
            "課程提供的 split 檔案目錄，包含 test_*.txt。"
            "例如：dataset/oxford-iiit-pet/nycu-2026-spring-dl-lab2-resnet34unet"
        )
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    inference(args)
