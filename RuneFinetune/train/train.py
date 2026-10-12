"""正式微调训练脚本。

- 设备自动选择：Intel 核显(xpu) -> CUDA -> CPU
- 用 model-0624.onnx 的权重初始化（全量微调），断点续训
- 损失：类别 BCE(pos_weight) 全锚点 + 关键点 conf BCE(pos_weight) 全锚点
        + 关键点 xy MSE（仅正样本锚点，权重 --kpt-weight）
- 目标分配：每个实例按 point_R（索引 2）所在网格单元分配到全部 3 个尺度
- 验证：损失 + 像素误差 / PCK@10 / PCK@20 / 类别命中率
        （单符帧用最高分锚点；多符帧每个对象贪心匹配同类别最近锚点）

用法示例：
  .venv-xpu/bin/python train/train.py \
      --train-labels data/labels.jsonl --val-labels data/val_labels.jsonl \
      --epochs 60 --batch 8
"""

import argparse
import functools
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model_recon"))
from import_weights import import_weights  # noqa: E402
from yolov5_rune import (  # noqa: E402
    NUM_CLASSES,
    NUM_KEYPOINTS,
    STRIDES,
    RuneYOLOv5,
    decode_outputs,
)

from dataset import RuneDataset, collate, load_labels  # noqa: E402

NET_W, NET_H = 640, 480


def pick_device(pref):
    if pref != "auto":
        if pref == "xpu" and not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            sys.exit("错误：--device xpu 但检测不到 Intel GPU。请先安装驱动：sudo apt install libze-intel-gpu1")
        return torch.device(pref)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_targets(objs_batch, device):
    """objs_batch: list[b] -> list[(cls_id, kpts [5,2] np)]"""
    b = len(objs_batch)
    targets = []
    for stride in STRIDES:
        h, w = NET_H // stride, NET_W // stride
        cls_t = torch.zeros(b, NUM_CLASSES, h, w, device=device)
        kpt_xy_t = torch.zeros(b, NUM_KEYPOINTS, 2, h, w, device=device)
        conf_t = torch.zeros(b, NUM_KEYPOINTS, h, w, device=device)
        mask = torch.zeros(b, NUM_KEYPOINTS, h, w, dtype=torch.bool, device=device)
        for bi, objs in enumerate(objs_batch):
            for cls_id, kpts in objs:
                kpts = torch.as_tensor(kpts, dtype=torch.float32, device=device)
                pr = kpts[2]
                gx = min(int(pr[0] / stride), w - 1)
                gy = min(int(pr[1] / stride), h - 1)
                raw = (kpts / stride - torch.tensor([gx, gy], dtype=torch.float32, device=device)) / 2.0
                cls_t[bi, cls_id, gy, gx] = 1.0
                kpt_xy_t[bi, :, :, gy, gx] = raw
                conf_t[bi, :, gy, gx] = 1.0
                mask[bi, :, gy, gx] = True
        targets.append((cls_t, kpt_xy_t, conf_t, mask))
    return targets


def compute_loss(outs, targets, pos_weight, kpt_weight):
    bce_cls = nn.BCEWithLogitsLoss(pos_weight=torch.full((NUM_CLASSES, 1, 1), pos_weight, device=outs[0].device))
    bce_conf = nn.BCEWithLogitsLoss(pos_weight=torch.full((NUM_KEYPOINTS, 1, 1), pos_weight, device=outs[0].device))
    mse = nn.MSELoss()
    total = torch.zeros((), device=outs[0].device)
    parts = {}
    for out, (cls_t, kpt_xy_t, conf_t, mask) in zip(outs, targets):
        cls_raw = out[:, :NUM_CLASSES]
        kpt_raw = out[:, NUM_CLASSES:].reshape(out.shape[0], NUM_KEYPOINTS, 3, *out.shape[2:])
        kpt_xy, kpt_conf = kpt_raw[:, :, :2], kpt_raw[:, :, 2]

        l_cls = bce_cls(cls_raw, cls_t)
        l_conf = bce_conf(kpt_conf, conf_t)
        mask_xy = mask.unsqueeze(2).expand_as(kpt_xy)
        l_kpt = mse(kpt_xy[mask_xy], kpt_xy_t[mask_xy])
        total = total + l_cls + l_conf + kpt_weight * l_kpt
        for k, v in (("cls", l_cls), ("conf", l_conf), ("kpt", l_kpt)):
            parts[k] = parts.get(k, 0.0) + v.item()
    return total / len(outs), parts


def decode_best(out):
    """out: [18,6300] 已解码。返回 (best_cls, quality, xy[5,2], kc[5])"""
    cls = out[:NUM_CLASSES]
    kpt = out[NUM_CLASSES:].reshape(NUM_KEYPOINTS, 3, -1)
    xy, kc = kpt[:, :2], kpt[:, 2]
    best_cls = torch.argmax(cls, dim=0)
    conf = cls[best_cls, torch.arange(cls.shape[1])]
    quality = conf * kc.mean(dim=0)
    i = int(torch.argmax(quality))
    return int(best_cls[i]), float(quality[i]), xy[:, :, i], kc[:, i]


def match_multi(dec_bi, objs, radius=32.0):
    """多对象帧：为每个 GT 对象在其 point_R 半径内找 quality 最高的预测锚点。

    与单符帧 decode_best 同口径（quality = cls_conf * mean(kpt_conf)），
    即"该符附近模型实际会响应的最强锚点"。
    返回 [(gt_cls, gt_kpts, pred_cls, xy[5,2])]，只含半径内存在候选的对象；
    调用方按 pred_cls == gt_cls 决定是否计入命中。
    """
    cls = dec_bi[:NUM_CLASSES]
    kpt = dec_bi[NUM_CLASSES:].reshape(NUM_KEYPOINTS, 3, -1)
    xy, kc = kpt[:, :2], kpt[:, 2]
    best_cls = torch.argmax(cls, dim=0)
    conf = cls[best_cls, torch.arange(cls.shape[1])]
    quality = conf * kc.mean(dim=0)
    pred_pr = xy[2]
    out = []
    for gi, (gt_cls, gt_kpts) in enumerate(objs):
        gt = torch.as_tensor(gt_kpts, dtype=torch.float32)
        d = (pred_pr - gt[2].unsqueeze(1)).norm(dim=0)
        q = quality.clone()
        q[d > radius] = -1.0
        if (q > -1.0).sum() == 0:
            continue
        j = int(torch.argmax(q))
        out.append((gt_cls, gt_kpts, int(best_cls[j]), xy[:, :, j]))
    return out


def evaluate(model, loader, device, pos_weight, kpt_weight):
    model.eval()
    loss_sum = {"cls": 0.0, "conf": 0.0, "kpt": 0.0}
    px_errs, pck10, pck20, cls_hit, n_objs, n_matched = [], 0, 0, 0, 0, 0
    with torch.no_grad():
        for xs, objs in loader:
            xs = xs.to(device)
            outs = model(xs)
            targets = build_targets(objs, device)
            loss, parts = compute_loss(outs, targets, pos_weight, kpt_weight)
            for k in loss_sum:
                loss_sum[k] += parts[k] * len(xs)
            dec = decode_outputs([o.cpu() for o in outs])
            for bi in range(len(xs)):
                objs_bi = objs[bi]
                if not objs_bi:
                    continue
                if len(objs_bi) == 1:
                    n_objs += 1
                    cls_id, _, xy, kc = decode_best(dec[bi])
                    gt_cls, gt_kpts = objs_bi[0]
                    if cls_id == gt_cls:
                        cls_hit += 1
                        n_matched += 1
                        d = (xy - torch.as_tensor(gt_kpts, dtype=torch.float32)).norm(dim=1)
                        px_errs.append(float(d.mean()))
                        pck10 += int((d < 10).sum())
                        pck20 += int((d < 20).sum())
                else:
                    n_objs += len(objs_bi)
                    for gt_cls, gt_kpts, pred_cls, xy in match_multi(dec[bi], objs_bi):
                        if pred_cls == gt_cls:
                            cls_hit += 1
                            n_matched += 1
                            d = (xy - torch.as_tensor(gt_kpts, dtype=torch.float32)).norm(dim=1)
                            px_errs.append(float(d.mean()))
                            pck10 += int((d < 10).sum())
                            pck20 += int((d < 20).sum())
    n = len(loader.dataset)
    mets = {
        "val_loss": sum(loss_sum.values()) / n,
        **{f"val_{k}": v / n for k, v in loss_sum.items()},
        "px_err": float(np.mean(px_errs)) if px_errs else -1.0,
        "pck10": pck10 / (n_matched * 5) if n_matched else -1.0,
        "pck20": pck20 / (n_matched * 5) if n_matched else -1.0,
        "cls_acc": cls_hit / n_objs if n_objs else -1.0,
        "n_objs": n_objs,
        "n_matched": n_matched,
    }
    return mets


def _worker_seed(seed, wid):
    np.random.seed(seed + wid)


def split_labels(labels_path, out_dir, val_frac, seed):
    recs = load_labels(labels_path)
    base = Path(labels_path).parent.resolve()
    for rec in recs:
        rec["file"] = str(base / rec["file"])
    idx = list(range(len(recs)))
    random.Random(seed).shuffle(idx)
    n_val = max(1, int(len(recs) * val_frac))
    tr, va = idx[n_val:], idx[:n_val]
    train_p = out_dir / "train_split.jsonl"
    val_p = out_dir / "val_split.jsonl"
    with open(train_p, "w") as f:
        for i in tr:
            f.write(json.dumps(recs[i]) + "\n")
    with open(val_p, "w") as f:
        for i in va:
            f.write(json.dumps(recs[i]) + "\n")
    return train_p, val_p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-labels", required=True)
    ap.add_argument("--val-labels", default=None, help="不传则从训练集随机划 10%%")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="auto", help="auto / xpu / cuda / cpu")
    ap.add_argument("--weights", default=str(ROOT / "weights" / "model-0624.onnx"), help="onnx 权重路径，或 '' 随机初始化")
    ap.add_argument("--out", default=None, help="输出目录，默认 train/runs/train_<时间戳>")
    ap.add_argument("--resume", default=None, help="从 last.pt 断点续训")
    ap.add_argument("--pos-weight", type=float, default=50.0)
    ap.add_argument("--kpt-weight", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4, help="CPU 时的线程数")
    ap.add_argument("--max-steps", type=int, default=None, help="冒烟测试用：每 epoch 最多跑几步")
    args = ap.parse_args()

    device = pick_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(args.threads)
    print(f"device: {device}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out) if args.out else ROOT / "train" / "runs" / f"train_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    val_labels = args.val_labels
    if val_labels is None:
        train_labels, val_labels = split_labels(args.train_labels, out_dir, 0.1, args.seed)
    else:
        train_labels = args.train_labels

    train_ds = RuneDataset(str(train_labels), augment=True, seed=args.seed)
    val_ds = RuneDataset(str(val_labels), augment=False)
    print(f"train: {len(train_ds)}  val: {len(val_ds)}")

    wi_fn = functools.partial(_worker_seed, args.seed)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                              collate_fn=collate, drop_last=True, worker_init_fn=wi_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                            collate_fn=collate, worker_init_fn=wi_fn)

    model = RuneYOLOv5().to(device)
    start_epoch = 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        start_epoch = ck.get("epoch", 0)
        print(f"resumed from {args.resume} (epoch {start_epoch})")
    elif args.weights:
        import_weights(model, args.weights)
        print(f"initialized from {args.weights}")
    else:
        print("WARNING: 随机初始化（未加载预训练权重）")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs), eta_min=args.lr * 0.05)
    if args.resume and Path(args.resume).with_name("opt.pt").exists():
        opt.load_state_dict(torch.load(Path(args.resume).with_name("opt.pt"), map_location="cpu", weights_only=False))

    best_loss = float("inf")
    metrics_log = []

    try:
        for epoch in range(start_epoch, args.epochs):
            model.train()
            t0 = time.time()
            run = {"cls": 0.0, "conf": 0.0, "kpt": 0.0}
            n_steps = 0
            for step, (xs, objs) in enumerate(train_loader):
                if args.max_steps and step >= args.max_steps:
                    break
                xs = xs.to(device)
                opt.zero_grad()
                outs = model(xs)
                targets = build_targets(objs, device)
                loss, parts = compute_loss(outs, targets, args.pos_weight, args.kpt_weight)
                loss.backward()
                opt.step()
                for k in run:
                    run[k] += parts[k]
                n_steps += 1
            sched.step()
            for k in run:
                run[k] /= max(1, n_steps)

            mets = evaluate(model, val_loader, device, args.pos_weight, args.kpt_weight)
            dt = time.time() - t0
            msg = (f"epoch {epoch + 1}/{args.epochs} | train loss {sum(run.values()):.4f} "
                   f"(cls {run['cls']:.4f} conf {run['conf']:.4f} kpt {run['kpt']:.4f}) | "
                   f"val loss {mets['val_loss']:.4f} | px {mets['px_err']:.1f} "
                   f"PCK10 {mets['pck10']:.3f} PCK20 {mets['pck20']:.3f} cls_acc {mets['cls_acc']:.3f} | {dt:.1f}s")
            print(msg)

            ck = {"model": model.state_dict(), "epoch": epoch + 1}
            torch.save(ck, out_dir / "last.pt")
            torch.save(opt.state_dict(), out_dir / "opt.pt")
            if mets["val_loss"] < best_loss:
                best_loss = mets["val_loss"]
                torch.save(ck, out_dir / "best.pt")
                print(f"  -> saved best.pt (val loss {best_loss:.4f})")
            metrics_log.append({"epoch": epoch + 1, "lr": sched.get_last_lr()[0], **run, **mets})
            with open(out_dir / "metrics.jsonl", "a") as f:
                f.write(json.dumps(metrics_log[-1]) + "\n")
    except KeyboardInterrupt:
        print(f"\ninterrupted, best val loss so far: {best_loss:.4f} (checkpoints in {out_dir})")

    print(f"done. outputs in {out_dir}")


if __name__ == "__main__":
    main()
