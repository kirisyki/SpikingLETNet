import os
from PIL import Image
import numpy as np

COLOR2ID = {
    (0, 0, 0): 0,          # Other
    (102, 102, 156): 1,    # Facade
    (128, 64, 128): 2,     # Road
    (107, 142, 35): 3,     # Vegetation
    (0, 0, 142): 4,        # Vehicle
    (70, 70, 70): 5        # Roof
}

def mask_rgb_to_class(mask: np.ndarray):
    h, w, _ = mask.shape
    label_mask = np.zeros((h, w), dtype=np.uint8)
    for rgb, cls_id in COLOR2ID.items():
        match = np.all(mask == rgb, axis=-1)
        label_mask[match] = cls_id
    return label_mask


def preprocess_and_save_overlap(txt_file, out_dir, list_out, patch_size=400, stride=200):
    """
    - patch_size: 每个patch大小 (H, W)
    - stride: 滑动窗口步长 (越小 -> 重叠越多)
    """
    base_dir = "/root/autodl-tmp/UDD/UDD6"
    os.makedirs(out_dir, exist_ok=True)
    out_f = open(list_out, "w")

    with open(txt_file, "r") as f:
        for line in f:
            img_path, mask_path = line.strip().split()
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            img_path = os.path.join(base_dir, img_path)
            mask_path = os.path.join(base_dir, mask_path)

            # 打开图像和mask
            image = np.array(Image.open(img_path).convert("RGB"))
            mask_rgb = np.array(Image.open(mask_path).convert("RGB"))
            mask = mask_rgb_to_class(mask_rgb)

            H, W = mask.shape

            # 计算所有左上角坐标
            y_starts = list(range(0, H - patch_size + 1, stride))
            x_starts = list(range(0, W - patch_size + 1, stride))

            # 保证覆盖到边缘
            if y_starts[-1] != H - patch_size:
                y_starts.append(H - patch_size)
            if x_starts[-1] != W - patch_size:
                x_starts.append(W - patch_size)

            for i in y_starts:
                for j in x_starts:
                    img_patch = image[i:i+patch_size, j:j+patch_size]
                    mask_patch = mask[i:i+patch_size, j:j+patch_size]
                    patch_id = f"{img_name}_{i}_{j}"

                    img_out = os.path.join(out_dir, f"{patch_id}_img.png")
                    mask_out = os.path.join(out_dir, f"{patch_id}_mask.png")

                    Image.fromarray(img_patch).save(img_out)
                    Image.fromarray(mask_patch).save(mask_out)

                    out_f.write(f"{img_out} {mask_out}\n")

    print(f"预处理完成，结果保存在 {out_dir}")


if __name__ == "__main__":
    preprocess_and_save_overlap("/root/autodl-tmp/UDD/UDD6/metadata/train.txt", "/root/autodl-tmp/UDD/UDD6/preprocessed/train_patches", "/root/autodl-tmp/UDD/UDD6/preprocessed/train_patches.txt", patch_size=400, stride=200)
    preprocess_and_save_overlap("/root/autodl-tmp/UDD/UDD6/metadata/val.txt", "/root/autodl-tmp/UDD/UDD6/preprocessed/val_patches", "/root/autodl-tmp/UDD/UDD6/preprocessed/val_patches.txt", patch_size=400, stride=200)