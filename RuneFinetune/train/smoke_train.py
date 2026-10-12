"""Smoke 训练：用几张图 + CPU 跑通训练闭环（数据加载→目标构建→损失→反向→参数更新）。

目标分配（暂定方案，正式训练前需确认）：
  - 每个实例按 point_R（索引 2）所在网格单元分配到全部 3 个尺度（P3/P4/P5）
  - 关键点回归目标使用与 ONNX decode 互逆的 raw 编码：raw = (px/stride - grid) / 2
  - 类别 BCE 只算分配锚点（+1）；关键点 conf BCE 只算分配锚点（+1）；xy 用 MSE
损失/负样本策略的完整版留到正式训练脚本再定。
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model_recon"))
from yolov5_rune import RuneYOLOv5, NUM_CLASSES, NUM_KEYPOINTS, STRIDES  # noqa: E402

# 14 线程混核 CPU 上 tiny op 线程颠簸会使前向慢 ~24 倍（实测 2.27s vs 0.096s）
torch.set_num_threads(4)

FRAMES_DIR = ROOT / "data" / "frames"
LABELS_PATH = ROOT / "data" / "smoke_labels.jsonl"
NET_W, NET_H = 640, 480
DEVICE = "cpu"


class SmokeDataset:
    def __init__(self):
        recs = [json.loads(l) for l in open(LABELS_PATH)]
        self.items = [(FRAMES_DIR / r["file"], r["objects"]) for r in recs]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, objects = self.items[i]
        img = Image.open(path).convert("RGB").resize((NET_W, NET_H), Image.LANCZOS)
        x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        kpts_list, cls_list = [], []
        for obj in objects:
            cls_list.append(obj["class_id"])
            kpts_list.append(torch.tensor(obj["keypoints"], dtype=torch.float32))  # [5,2]
        return x, cls_list, kpts_list


def build_targets(cls_list, kpts_list):
    """每尺度返回 (cls_t [1,3,H,W], kpt_xy_t [1,5,2,H,W], kpt_mask [1,5,H,W], conf_t [1,5,H,W])"""
    targets = []
    for stride in STRIDES:
        h, w = NET_H // stride, NET_W // stride
        cls_t = torch.zeros(1, NUM_CLASSES, h, w)
        kpt_xy_t = torch.zeros(1, NUM_KEYPOINTS, 2, h, w)
        conf_t = torch.zeros(1, NUM_KEYPOINTS, h, w)
        mask = torch.zeros(1, NUM_KEYPOINTS, h, w, dtype=torch.bool)
        for cls_id, kpts in zip(cls_list, kpts_list):
            pr = kpts[2]  # point_R 决定分配网格
            gx = int(pr[0] / stride)
            gy = int(pr[1] / stride)
            gx = min(gx, w - 1)
            gy = min(gy, h - 1)
            raw = (kpts / stride - torch.tensor([gx, gy], dtype=torch.float32)) / 2.0
            cls_t[0, cls_id, gy, gx] = 1.0
            kpt_xy_t[0, :, :, gy, gx] = raw
            conf_t[0, :, gy, gx] = 1.0
            mask[0, :, gy, gx] = True
        targets.append((cls_t, kpt_xy_t, mask, conf_t))
    return targets


def compute_loss(outs, targets):
    """outs: 3 个尺度原始头输出 [B,18,H,W]；targets 由 build_targets 生成"""
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    total = torch.zeros((), device=DEVICE)
    parts = {}
    for out, (cls_t, kpt_xy_t, mask, conf_t) in zip(outs, targets):
        cls_raw = out[:, :NUM_CLASSES]
        kpt_raw = out[:, NUM_CLASSES:].reshape(1, NUM_KEYPOINTS, 3, *cls_t.shape[2:])
        kpt_xy = kpt_raw[:, :, :2]
        kpt_conf = kpt_raw[:, :, 2]

        l_cls = bce(cls_raw[:, :, mask[0, 0]], cls_t[:, :, mask[0, 0]])
        mask_xy = mask.unsqueeze(2).expand_as(kpt_xy)
        l_kpt = mse(kpt_xy[mask_xy], kpt_xy_t[mask_xy])
        l_conf = bce(kpt_conf[mask], conf_t[mask])
        total = total + l_cls + 10.0 * l_kpt + l_conf
        parts.setdefault("cls", []).append(l_cls.item())
        parts.setdefault("kpt", []).append(l_kpt.item())
        parts.setdefault("conf", []).append(l_conf.item())
    return total, {k: float(np.mean(v)) for k, v in parts.items()}


def decode_assigned(outs, kpts_list):
    """把每个实例分配锚点的预测 xy 解码回像素，用于看回归是否在变好。"""
    preds, gts = [], []
    for stride, out in zip(STRIDES, outs):
        kpt_raw = out[0, NUM_CLASSES:].reshape(NUM_KEYPOINTS, 3, *out.shape[2:])
        kxy = kpt_raw[:, :2]
        for kpts in kpts_list:
            pr = kpts[2]
            gx = min(int(pr[0] / stride), out.shape[3] - 1)
            gy = min(int(pr[1] / stride), out.shape[2] - 1)
            pred = (kxy[:, :, gy, gx] * 2.0 + torch.tensor([gx, gy], dtype=torch.float32)) * stride
            preds.append(pred)
            gts.append(kpts)
    return preds, gts


def main():
    torch.manual_seed(0)
    ds = SmokeDataset()
    model = RuneYOLOv5().to(DEVICE)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    steps = 30
    print(f"images={len(ds)} steps={steps} device={DEVICE} params={sum(p.numel() for p in model.parameters())}")

    first_losses = None
    for step in range(steps):
        opt.zero_grad()
        losses_sum = {}
        total = torch.zeros((), device=DEVICE)
        for i in range(len(ds)):
            x, cls_list, kpts_list = ds[i]
            outs = model(x.unsqueeze(0))
            targets = build_targets(cls_list, kpts_list)
            loss, parts = compute_loss(outs, targets)
            total = total + loss
            for k, v in parts.items():
                losses_sum[k] = losses_sum.get(k, 0.0) + v
        if first_losses is None:
            first_losses = {k: v / len(ds) for k, v in losses_sum.items()}
        total = total / len(ds)
        total.backward()
        if step == 0:
            gnorm = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
            print(f"step 0: grad abs sum={gnorm:.3e}")
        opt.step()
        if step % 5 == 0 or step == steps - 1:
            print(f"step {step:2d}: total={total.item():.4f} " +
                  " ".join(f"{k}={losses_sum[k]/len(ds):.4f}" for k in sorted(losses_sum)))

    print("first losses:", first_losses)

    # 训练前后：分配锚点预测关键点 vs 标签的像素误差
    model.eval()
    model0 = RuneYOLOv5().to(DEVICE)
    model0.eval()
    for name, m in [("init", model0), ("final", model)]:
        errs = []
        with torch.no_grad():
            for i in range(len(ds)):
                x, cls_list, kpts_list = ds[i]
                outs = m(x.unsqueeze(0))
                preds, gts = decode_assigned(outs, kpts_list)
                errs += [float((p - g).abs().mean()) for p, g in zip(preds, gts)]
        print(f"{name}: mean |pred-gt| px = {np.mean(errs):.3f} (per anchor-instance)")

    ckpt = ROOT / "train" / "smoke_ckpt.pt"
    torch.save(model.state_dict(), ckpt)
    print(f"saved {ckpt}")


if __name__ == "__main__":
    main()
