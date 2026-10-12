"""训练数据集：PNG 帧 + JSONL 标签（坐标在 640x480 网络输入空间）。

标签格式（每行一条）：
  {"file": "frame_xxx.png", "objects": [{"class_id": 0, "keypoints": [[x,y] x5]}]}
file 相对标签文件所在目录解析。

关键点顺序（已与用户确认）：0=top(绿) 1=left(黄) 2=point_R(蓝) 3=right(品红) 4=bottom(红)
水平翻转：叶片/亮叶交换 1<->3；三叉戟（class 2）交换 1<->3 与 0<->4。
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

NET_W, NET_H = 640, 480


def load_labels(path):
    recs = [json.loads(line) for line in open(path) if line.strip()]
    return recs


def hsv_jitter(img_rgb, h_gain=0.015, s_gain=0.4, v_gain=0.4):
    """YOLOv5 风格 HSV 扰动，img_rgb: [H,W,3] uint8"""
    x = np.asarray(img_rgb, dtype=np.float32) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    maxv = x.max(axis=2)
    minv = x.min(axis=2)
    v = maxv
    s = np.where(maxv > 0, (maxv - minv) / np.maximum(maxv, 1e-7), 0.0)
    h = np.zeros_like(v)
    # hue 计算
    d = maxv - minv
    idx = d > 1e-7
    cr = (maxv - r) / np.maximum(d, 1e-7)
    cg = (maxv - g) / np.maximum(d, 1e-7)
    cb = (maxv - b) / np.maximum(d, 1e-7)
    h = np.where(idx & (maxv == r), cb - cg, h)
    h = np.where(idx & (maxv == g), 2.0 + cr - cb, h)
    h = np.where(idx & (maxv == b), 4.0 + cg - cr, h)
    h = (h / 6.0) % 1.0
    # 扰动
    h = (h + random.uniform(-h_gain, h_gain)) % 1.0
    s = np.clip(s * random.uniform(1 - s_gain, 1 + s_gain), 0, 1)
    v = np.clip(v * random.uniform(1 - v_gain, 1 + v_gain), 0, 1)
    # HSV -> RGB
    i = (h * 6.0).astype(np.int32) % 6
    f = h * 6.0 - i.astype(np.float32)
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    i %= 6
    out = np.stack(
        [
            np.where(i == 0, v, np.where(i == 1, q, np.where(i == 2, p, np.where(i == 3, p, np.where(i == 4, t, v))))),
            np.where(i == 0, q, np.where(i == 1, v, np.where(i == 2, v, np.where(i == 3, t, np.where(i == 4, p, p))))),
            np.where(i == 0, p, np.where(i == 1, p, np.where(i == 2, q, np.where(i == 3, v, np.where(i == 4, v, t))))),
        ],
        axis=-1,
    )
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def apply_flip(img_rgb, objects):
    """水平翻转；关键点 x -> NET_W-1-x。

    镜像索引交换：叶片/亮叶 1(left)<->3(right)；三叉戟（class 2）横杠两端 1<->3、叉尖两端 0<->4。
    """
    img_rgb = img_rgb[:, ::-1].copy()
    out_objs = []
    for cls_id, kpts in objects:
        kpts = kpts.copy()
        kpts[:, 0] = NET_W - 1 - kpts[:, 0]
        kpts[[1, 3]] = kpts[[3, 1]]
        if cls_id == 2:
            kpts[[0, 4]] = kpts[[4, 0]]
        out_objs.append((cls_id, kpts))
    return img_rgb, out_objs


class RuneDataset(torch.utils.data.Dataset):
    def __init__(self, labels_path, augment=False, seed=None):
        self.recs = load_labels(labels_path)
        self.base_dir = Path(labels_path).parent
        self.augment = augment
        self.rng = random.Random(seed)
        self.items = []
        for rec in self.recs:
            objs = []
            for obj in rec["objects"]:
                kpts = np.asarray(obj["keypoints"], dtype=np.float32).reshape(5, 2)
                objs.append((int(obj["class_id"]), kpts))
            self.items.append((self.base_dir / rec["file"], objs))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, objects = self.items[i]
        img = Image.open(path).convert("RGB").resize((NET_W, NET_H), Image.LANCZOS)
        img_rgb = np.asarray(img, dtype=np.uint8)
        if self.augment:
            img_rgb = hsv_jitter(img_rgb)
            if self.rng.random() < 0.5:
                img_rgb, objects = apply_flip(img_rgb, objects)
        x = torch.from_numpy(img_rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
        return x, objects


def collate(batch):
    xs = torch.stack([b[0] for b in batch])
    objs = [b[1] for b in batch]
    return xs, objs
