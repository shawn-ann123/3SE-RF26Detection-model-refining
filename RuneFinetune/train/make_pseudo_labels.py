"""用预训练 ONNX 在采集帧上跑推理，把最佳检测写成 JSONL 伪标签（480x640 空间）。

目的：给 smoke 训练程序提供最小标签数据（跑通逻辑用，非正式标注）。
标签格式（与后续正式标注流水线一致）：
  {"file": "frame_xxx.png", "objects": [{"class_id": int, "keypoints": [[x,y] x5]}]}
"""

import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
FRAMES_DIR = ROOT / "data" / "frames"
OUT_LABELS = ROOT / "data" / "smoke_labels.jsonl"
ONNX_PATH = ROOT / "weights" / "model-0624.onnx"

NET_W, NET_H = 640, 480
KPT_COLORS = [(0, 255, 0), (0, 255, 255), (255, 0, 0), (255, 0, 255), (0, 0, 255)]


def load_image_rgb(path):
    img = Image.open(path).convert("RGB").resize((NET_W, NET_H), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))[None]  # [1,3,H,W]


def decode_best(onnx_out):
    """onnx_out [1,18,6300]（已解码）。返回 (class_id, quality, kpts[5][x,y], confs)"""
    out = onnx_out[0]  # [18, 6300]
    cls = out[:3]
    kpt = out[3:].reshape(5, 3, -1)  # [5, 3(x,y,c), 6300]
    xy, kc = kpt[:, :2], kpt[:, 2]
    best_cls = np.argmax(cls, axis=0)
    conf = cls[best_cls, np.arange(cls.shape[1])]
    mean_kc = kc.mean(axis=0)
    quality = conf * mean_kc
    i = int(np.argmax(quality))
    return int(best_cls[i]), float(quality[i]), xy[:, :, i], kc[:, i]


def annotate(path, kpts, kpt_colors):
    img = Image.open(path).convert("RGB").resize((NET_W, NET_H), Image.LANCZOS)
    d = ImageDraw.Draw(img)
    for i, (x, y) in enumerate(kpts):
        r = 5
        d.ellipse([x - r, y - r, x + r, y + r], outline=kpt_colors[i], width=2)
        d.text((x + 6, y - 6), str(i), fill=kpt_colors[i])
    return img


def main():
    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    pngs = sorted(FRAMES_DIR.glob("*.png"))
    print(f"frames: {len(pngs)}")

    labels = []
    for png in pngs:
        arr = load_image_rgb(png)
        out = sess.run(None, {"images": arr})[0]
        cls_id, quality, xy, kc = decode_best(out)
        kpts = [[round(float(x), 1), round(float(y), 1)] for x, y in xy]
        print(f"{png.name}: class={cls_id} quality={quality:.3f} kpt_conf={kc}")
        print("  kpts:", kpts)
        labels.append({"file": str(Path("frames") / png.name), "objects": [{"class_id": cls_id, "keypoints": kpts}]})

        ann = annotate(png, np.array(kpts), KPT_COLORS)
        ann.save(f"/tmp/anno_{png.stem}.png")

    with open(OUT_LABELS, "w") as f:
        for rec in labels:
            f.write(json.dumps(rec) + "\n")
    print(f"wrote {OUT_LABELS}")


if __name__ == "__main__":
    main()
