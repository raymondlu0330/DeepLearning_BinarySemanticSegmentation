"""一行指令完成 Train → Evaluate → Inference 的整合腳本。

使用方式（UNet）：
    python src/run.py --model unet --epochs 100 --batch_size 32 --lr 1e-4 --gpu 0 \\
        --list_dir dataset/oxford-iiit-pet/nycu-2026-spring-dl-lab2-unet

使用方式（ResNet34-UNet）：
    python src/run.py --model resnet34_unet --epochs 50 --batch_size 16 --lr 5e-4 --gpu 0 \\
        --list_dir dataset/oxford-iiit-pet/nycu-2026-spring-dl-lab2-resnet34unet

只跑 Evaluate + Inference（跳過訓練，指定既有 checkpoint）：
    python src/run.py --model unet --skip_train \\
        --model_path saved_models/unet_0324_1855_best.pth --gpu 0 \\
        --list_dir dataset/oxford-iiit-pet/nycu-2026-spring-dl-lab2-unet
"""

import os
import sys
import argparse

# 讓 Python 能找到同目錄下的其他模組
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ─────────────────────────────────────────────────────────────
# Step 1: Train
# ─────────────────────────────────────────────────────────────
def run_train(args):
    """執行訓練，回傳最佳 checkpoint 路徑。"""
    import torch
    from datetime import datetime
    from torch.optim import Adam
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from oxford_pet import OxfordPetDataset
    from utils import dice_score, combined_loss, save_checkpoint
    from models.unet import UNet
    from models.resnet34_unet import ResNet34UNet

    def get_model(name):
        if name == "unet":
            return UNet()
        elif name == "resnet34_unet":
            return ResNet34UNet()
        raise ValueError(f"Unknown model: {name}")

    def train_one_epoch(model, loader, optimizer, device, bce_weight):
        model.train()
        total_loss, total_dice = 0.0, 0.0
        for images, masks in tqdm(loader, desc="Train", leave=False):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            preds = model(images)
            loss = combined_loss(preds, masks, bce_weight=bce_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            total_dice += dice_score(preds.detach(), masks.detach())
        n = len(loader)
        return total_loss / n, total_dice / n

    @torch.no_grad()
    def validate(model, loader, device, bce_weight):
        model.eval()
        total_loss, total_dice = 0.0, 0.0
        for images, masks in tqdm(loader, desc="Val  ", leave=False):
            images, masks = images.to(device), masks.to(device)
            preds = model(images)
            loss = combined_loss(preds, masks, bce_weight=bce_weight)
            total_loss += loss.item()
            total_dice += dice_score(preds, masks)
        n = len(loader)
        return total_loss / n, total_dice / n

    # ---- 裝置 ----
    if args.gpu is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"[TRAIN] 模型: {args.model}  裝置: {device}")
    print(f"{'='*60}")

    # ---- DataLoader ----
    train_ds = OxfordPetDataset(args.data_path, mode="train",
                                img_size=args.img_size, list_dir=args.list_dir)
    val_ds   = OxfordPetDataset(args.data_path, mode="val",
                                img_size=args.img_size, list_dir=args.list_dir)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    print(f"Train: {len(train_ds)} 張 | Val: {len(val_ds)} 張")

    # ---- 模型、優化器、排程器 ----
    model     = get_model(args.model).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ---- Checkpoint 路徑 ----
    os.makedirs(args.save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%m%d_%H%M")
    best_path = os.path.join(args.save_dir, f"{args.model}_{timestamp}_best.pth")

    best_dice = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_dice = train_one_epoch(
            model, train_loader, optimizer, device, args.bce_weight)
        val_loss, val_dice = validate(
            model, val_loader, device, args.bce_weight)
        scheduler.step()

        print(
            f"Epoch [{epoch:>3}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f}  Train Dice: {train_dice:.4f}  "
            f"Val Loss: {val_loss:.4f}  Val Dice: {val_dice:.4f}"
        )
        if val_dice > best_dice:
            best_dice = val_dice
            save_checkpoint(model, optimizer, epoch, val_dice, best_path)
            print(f"  -> 儲存最佳模型 (Val Dice: {best_dice:.4f})")

    print(f"\n[TRAIN] 完成。Best Val Dice: {best_dice:.4f}")
    print(f"[TRAIN] 最佳 checkpoint: {best_path}")
    return best_path


# ─────────────────────────────────────────────────────────────
# Step 2: Evaluate
# ─────────────────────────────────────────────────────────────
def run_evaluate(args, model_path):
    """在驗證集上評估指定 checkpoint，印出 Dice Score。"""
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from oxford_pet import OxfordPetDataset
    from utils import dice_score, load_checkpoint
    from models.unet import UNet
    from models.resnet34_unet import ResNet34UNet

    def get_model(name):
        if name == "unet":
            return UNet()
        elif name == "resnet34_unet":
            return ResNet34UNet()
        raise ValueError(f"Unknown model: {name}")

    if args.gpu is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"[EVALUATE] 模型: {args.model}  Checkpoint: {model_path}")
    print(f"{'='*60}")

    val_ds = OxfordPetDataset(args.data_path, mode="val",
                              img_size=args.img_size, list_dir=args.list_dir)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = get_model(args.model).to(device)
    ckpt  = load_checkpoint(model, model_path, device)
    saved_epoch = ckpt.get("epoch", "?")
    saved_dice  = ckpt.get("val_dice", float("nan"))
    print(f"Checkpoint: epoch={saved_epoch}, 儲存時 Val Dice={saved_dice:.4f}")
    model.eval()

    all_dice = []
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Evaluating"):
            images, masks = images.to(device), masks.to(device)
            preds = model(images)
            all_dice.append(dice_score(preds, masks))

    mean_dice = sum(all_dice) / len(all_dice)
    print(f"\n[EVALUATE] 驗證集平均 Dice Score: {mean_dice:.4f}")
    return mean_dice


# ─────────────────────────────────────────────────────────────
# Step 3: Inference
# ─────────────────────────────────────────────────────────────
def run_inference(args, model_path):
    """對測試集推論，輸出 Kaggle CSV。"""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from PIL import Image

    from oxford_pet import OxfordPetDataset
    from utils import rle_encode, load_checkpoint
    from models.unet import UNet
    from models.resnet34_unet import ResNet34UNet

    def get_model(name):
        if name == "unet":
            return UNet()
        elif name == "resnet34_unet":
            return ResNet34UNet()
        raise ValueError(f"Unknown model: {name}")

    if args.gpu is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"[INFERENCE] 模型: {args.model}  Checkpoint: {model_path}")
    print(f"{'='*60}")

    test_ds = OxfordPetDataset(args.data_path, mode="test",
                               img_size=args.img_size, list_dir=args.list_dir)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    print(f"測試集圖片數量: {len(test_ds)}")

    model = get_model(args.model).to(device)
    load_checkpoint(model, model_path, device)
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, f"{args.model}_submission.csv")
    rows = []

    with torch.no_grad():
        for images, names, orig_ws, orig_hs in tqdm(test_loader, desc="Inference"):
            images = images.to(device)
            preds  = model(images)
            preds_bin = (preds > args.threshold).squeeze(1).cpu().numpy().astype(np.uint8)

            for pred_mask, name, orig_w, orig_h in zip(preds_bin, names, orig_ws, orig_hs):
                orig_w, orig_h = int(orig_w), int(orig_h)
                if pred_mask.shape != (orig_h, orig_w):
                    pil_mask  = Image.fromarray(pred_mask).resize((orig_w, orig_h), Image.NEAREST)
                    pred_mask = np.array(pil_mask, dtype=np.uint8)
                rows.append((name, rle_encode(pred_mask)))

    with open(csv_path, "w") as f:
        f.write("image_id,encoded_mask\n")
        for image_id, rle in rows:
            f.write(f"{image_id},{rle}\n")

    print(f"\n[INFERENCE] 完成，共 {len(rows)} 張圖片。")
    print(f"[INFERENCE] Kaggle CSV 儲存至: {csv_path}")
    return csv_path


# ─────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="一行指令執行 Train → Evaluate → Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # 模型
    parser.add_argument("--model", type=str, required=True,
                        choices=["unet", "resnet34_unet"], help="模型架構")

    # 跳過訓練（直接用既有 checkpoint）
    parser.add_argument("--skip_train", action="store_true",
                        help="跳過訓練，直接用 --model_path 做 evaluate + inference")
    parser.add_argument("--model_path", type=str, default=None,
                        help="指定既有 checkpoint（搭配 --skip_train 使用）")

    # 路徑
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--data_path",  type=str,
                        default=os.path.join(_root, "dataset", "oxford-iiit-pet"))
    parser.add_argument("--save_dir",   type=str,
                        default=os.path.join(_root, "saved_models"))
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(_root, "predictions"))
    parser.add_argument("--list_dir",   type=str, default=None,
                        help="課程提供的 split 目錄（含 train.txt / val.txt / test_*.txt）")

    # 訓練超參數
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch_size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--bce_weight", type=float, default=0.5,
                        help="combined_loss 中 BCE 的權重（Dice = 1 - bce_weight）")
    parser.add_argument("--img_size",   type=int,   default=256)
    parser.add_argument("--num_workers",type=int,   default=4)
    parser.add_argument("--gpu",        type=int,   default=None,
                        help="GPU 編號，不指定時自動選擇")

    # Inference 專用
    parser.add_argument("--threshold",  type=float, default=0.5,
                        help="二值化閾值")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # ── Step 1: Train（或跳過）──
    if args.skip_train:
        if args.model_path is None:
            raise ValueError("使用 --skip_train 時必須同時指定 --model_path")
        best_path = args.model_path
        print(f"[RUN] 跳過訓練，使用既有 checkpoint: {best_path}")
    else:
        best_path = run_train(args)

    # ── Step 2: Evaluate ──
    run_evaluate(args, best_path)

    # ── Step 3: Inference ──
    run_inference(args, best_path)

    print(f"\n{'='*60}")
    print(f"[RUN] 全部完成！")
    print(f"  Checkpoint : {best_path}")
    print(f"  CSV        : {os.path.join(args.output_dir, args.model + '_submission.csv')}")
    print(f"{'='*60}")
