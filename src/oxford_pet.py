"""Oxford-IIIT Pet 資料集的 PyTorch Dataset 定義。

Trimap 標注說明：
    pixel 值 1 → 前景（寵物本體），轉換為 binary mask 的 1.0
    pixel 值 2 → 背景，轉換為 binary mask 的 0.0
    pixel 值 3 → 邊界（未分類），視為背景，轉換為 0.0

【本次新增】
    - list_dir 參數：支援讀取課程提供的指定 split 檔案
      （nycu-2026-spring-dl-lab2-unet / nycu-2026-spring-dl-lab2-resnet34unet）
    - test mode 額外回傳 orig_w, orig_h：供 inference.py 將 mask resize 回原始尺寸
"""

import os
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision import transforms


class OxfordPetDataset(Dataset):
    def __init__(self, root, mode="train", img_size=256, list_dir=None):
        """
        Args:
            root (str):      資料集根目錄，須包含 images/ 和 annotations/ 子目錄。
            mode (str):      'train'、'val' 或 'test' 三選一。
            img_size (int):  將圖片和 mask resize 到此正方形大小（預設 256）。
            list_dir (str):  【新增】課程提供的 split 檔案目錄路徑。
                             提供時直接讀取該目錄下的 train.txt / val.txt / test_*.txt，
                             不再使用原始的 trainval.txt 80/20 切分。
                             例如：'dataset/oxford-iiit-pet/nycu-2026-spring-dl-lab2-unet'
        """
        assert mode in ("train", "val", "test"), f"無效的 mode: {mode}"
        self.root = root
        self.mode = mode
        self.img_size = img_size

        # 圖片和 trimap mask 的路徑
        self.image_dir = os.path.join(root, "images")
        self.mask_dir = os.path.join(root, "annotations", "trimaps")

        if list_dir is not None:
            # ---- 【新增】使用課程提供的指定 split 檔案 ----
            if mode == "train":
                list_file = os.path.join(list_dir, "train.txt")
            elif mode == "val":
                list_file = os.path.join(list_dir, "val.txt")
            else:
                # test 模式：自動搜尋 test_*.txt
                # （UNet 用 test_unet.txt；ResNet34 用 test_res_unet.txt）
                import glob
                candidates = glob.glob(os.path.join(list_dir, "test*.txt"))
                if not candidates:
                    raise FileNotFoundError(f"找不到 test*.txt 於 {list_dir}")
                list_file = sorted(candidates)[0]  # 取第一個匹配的檔案

            # 讀取檔名清單（每行第一個 token 為檔名，忽略 # 開頭的註解行）
            with open(list_file) as f:
                self.filenames = [
                    line.strip().split()[0]
                    for line in f
                    if line.strip() and not line.startswith("#")
                ]
        else:
            # ---- 原始行為：讀 trainval.txt 並做 80/20 切分 ----
            if mode in ("train", "val"):
                list_file = os.path.join(root, "annotations", "trainval.txt")
            else:
                list_file = os.path.join(root, "annotations", "test.txt")

            with open(list_file) as f:
                lines = [
                    line.strip().split()[0]
                    for line in f
                    if line.strip() and not line.startswith("#")
                ]

            # 80/20 切分，使用固定隨機種子 42 確保可重現
            if mode in ("train", "val"):
                rng = random.Random(42)
                indices = list(range(len(lines)))
                rng.shuffle(indices)
                split = int(0.8 * len(lines))
                if mode == "train":
                    self.filenames = [lines[i] for i in indices[:split]]
                else:
                    self.filenames = [lines[i] for i in indices[split:]]
            else:
                self.filenames = lines

        print(f'[{self.mode}] 載入 {len(self.filenames)} 張圖片')

        # ImageNet 的 RGB 均值和標準差，用於輸入標準化
        self.mean = [0.485, 0.456, 0.406]  # RGB 三通道均值
        self.std  = [0.229, 0.224, 0.225]  # RGB 三通道標準差

    def __len__(self):
        return len(self.filenames)

    def _load_image(self, name):
        """載入 RGB 圖片。"""
        path = os.path.join(self.image_dir, name + ".jpg")
        return Image.open(path).convert("RGB")

    def _load_mask(self, name):
        """載入 Trimap 標注（pixel 值 1/2/3）。"""
        path = os.path.join(self.mask_dir, name + ".png")
        return Image.open(path)

    def _augment(self, image, mask):
        """訓練集資料增強（幾何變換同時套用於 image 和 mask；色彩變換只套用於 image）。"""

        # 隨機水平翻轉（p=0.5）
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask  = TF.hflip(mask)

        # 隨機垂直翻轉（p=0.5）
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask  = TF.vflip(mask)

        # 隨機旋轉 ±15°（p=0.5）
        # image 用 bilinear 插值；mask 用 nearest 避免引入非整數 label
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
            mask  = TF.rotate(mask,  angle, interpolation=TF.InterpolationMode.NEAREST)

        # 隨機縮放裁切並 resize 回目標大小（p=0.5）
        # scale=(0.7, 1.0)：裁切面積為原圖 70%~100%
        # ratio=(0.75, 1.33)：裁切區域的長寬比範圍
        if random.random() > 0.5:
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                image, scale=(0.7, 1.0), ratio=(0.75, 1.33)
            )
            image = TF.resized_crop(
                image, i, j, h, w, (self.img_size, self.img_size),
                interpolation=TF.InterpolationMode.BILINEAR
            )
            mask = TF.resized_crop(
                mask, i, j, h, w, (self.img_size, self.img_size),
                interpolation=TF.InterpolationMode.NEAREST
            )

        # Color Jitter（p=1.0，只對 image，mask 不動）
        # 亮度、對比度、飽和度各 ±30%，色調 ±5%
        color_jitter = transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05
        )
        image = color_jitter(image)

        return image, mask

    def __getitem__(self, idx):
        name  = self.filenames[idx]
        image = self._load_image(name)

        if self.mode == "test":
            # 【新增】記錄原始圖片尺寸（W, H），inference 時將 mask resize 回原始大小
            # Kaggle 用原始圖片尺寸解碼 RLE，所以 mask 必須是原始尺寸
            orig_w, orig_h = image.size
            image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
            image = TF.to_tensor(image)
            image = TF.normalize(image, self.mean, self.std)
            # 回傳 (image_tensor, 檔名, 原始寬, 原始高)
            return image, name, orig_w, orig_h

        # ---- Train / Val 模式 ----

        # Resize 圖片到目標大小
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)

        # 載入並 resize mask（使用 NEAREST 避免插值產生非整數值）
        mask = self._load_mask(name)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        # 訓練集做資料增強；驗證集不做
        if self.mode == "train":
            image, mask = self._augment(image, mask)

        # 將 image 轉為 Tensor 並用 ImageNet 統計做標準化
        image = TF.to_tensor(image)
        image = TF.normalize(image, self.mean, self.std)

        # 將 Trimap 轉換為 binary mask
        # pixel==1 → 前景（1.0）；其他（2=背景、3=邊界）→ 背景（0.0）
        mask_np     = np.array(mask, dtype=np.float32)
        binary_mask = (mask_np == 1).astype(np.float32)
        # unsqueeze(0) 加上 channel 維度 → (1, H, W)
        mask_tensor = torch.from_numpy(binary_mask).unsqueeze(0)

        return image, mask_tensor


def get_dataloaders(data_root, img_size=256, batch_size=16, num_workers=4):
    """建立 train、val、test 三個 DataLoader 的便利函式。"""
    from torch.utils.data import DataLoader

    train_ds = OxfordPetDataset(data_root, mode="train", img_size=img_size)
    val_ds   = OxfordPetDataset(data_root, mode="val",   img_size=img_size)
    test_ds  = OxfordPetDataset(data_root, mode="test",  img_size=img_size)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # 快速測試：確認 DataLoader 正常運作
    train_loader, val_loader, test_loader = get_dataloaders(
        "dataset/oxford-iiit-pet"
    )
    images, masks = next(iter(train_loader))
    print("圖片 shape:", images.shape)   # (B, 3, 256, 256)
    print("Mask shape:", masks.shape)    # (B, 1, 256, 256)
