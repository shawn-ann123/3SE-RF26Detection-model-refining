"""PyTorch 重建：RuneDetectionModel 五点能量机关检测网络。

架构为 YOLOv5n 轻量剪裁版（通道减半），输入 [B,3,480,640]，
检测头输出 3 类 + 5 关键点 (x,y,conf) = 18 通道。

与 model-0624.onnx 的逐层对应（权重名可直接映射）：
  model.0  Conv 3->16  k3 s2
  model.1  Conv 16->32 k3 s2
  model.2  C3(32->32, n=1)
  model.3  Conv 32->64 k3 s2
  model.4  C3(64->64, n=2)
  model.5  Conv 64->128 k3 s2
  model.6  C3(128->128, n=2)
  model.7  Conv 128->256 k3 s2
  model.8  C3(256->256, n=1)
  model.9  SPPF(256->256, cv1 256->128)
  model.10 Resize x2
  model.11 Concat(256+128=384)
  model.12 C3(384->128, n=1, shortcut=False)
  model.13 Resize x2
  model.14 Concat(128+64=192)
  model.15 C3(192->64, n=1, shortcut=False)   -> P3/8
  model.16 Conv 64->64 k3 s2
  model.17 Concat(64+128=192)
  model.18 C3(192->128, n=1, shortcut=False)  -> P4/16
  model.19 Conv 128->128 k3 s2
  model.20 Concat(128+256=384)
  model.21 C3(384->256, n=1, shortcut=False)  -> P5/32
  model.22 Detect: 每层 cv3(in->64->64->3 类) + cv4(in->48->48->15 关键点)
"""

import math

import torch
import torch.nn as nn

NUM_CLASSES = 3
NUM_KEYPOINTS = 5
KPT_DIM = 3  # x, y, conf
STRIDES = (8, 16, 32)


def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """conv + SiLU。ONNX 导出时 BN 已融合进 conv（带 bias、无 BN 参数），故此处不加 BN。"""

    def __init__(self, c1, c2, k=1, s=1, p=None):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), bias=True)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    def __init__(self, c, shortcut=True):
        super().__init__()
        self.cv1 = Conv(c, c, 3)
        self.cv2 = Conv(c, c, 3)
        self.shortcut = shortcut

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.shortcut else y


class C3(nn.Module):
    """与 ONNX 导出的 C3 变体一致：
    cv1(1x1, c1->c2) -> Split[c2/2, c2/2] -> 后半过 n 个 3x3 瓶颈(可残差)
    -> Concat[前半, 后半, 各瓶颈输出] -> cv2(1x1, (n+2)*c2/2 -> c2)
    """

    def __init__(self, c1, c2, n=1, shortcut=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c2, 1)
        self.m = nn.ModuleList(Bottleneck(c_, shortcut) for _ in range(n))
        self.cv2 = Conv(c_ * (n + 2), c2, 1)

    def forward(self, x):
        y = self.cv1(x)
        a, b = y.split(y.size(1) // 2, dim=1)
        outs = [a, b]
        cur = b
        for m in self.m:
            cur = m(cur)
            outs.append(cur)
        return self.cv2(torch.cat(outs, dim=1))


class SPPF(nn.Module):
    def __init__(self, c1, c2, c_=None):
        super().__init__()
        c_ = c_ or c1 // 2
        self.cv1 = Conv(c1, c_, 1)
        self.cv2 = Conv(c_ * 4, c2, 1)
        self.m = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat((x, y1, y2, y3), dim=1))


class DetectHead(nn.Module):
    """检测头：3 类 + 5 关键点 (x,y,conf)，训练模式返回原始卷积输出。"""

    def __init__(self, in_channels=(64, 128, 256), nc=3, nk=5):
        super().__init__()
        self.nc = nc
        self.nk = nk
        self.cv3 = nn.ModuleList()  # 类别头 in->64->64->nc
        self.cv4 = nn.ModuleList()  # 关键点头 in->48->48->nk*3
        for ch in in_channels:
            self.cv3.append(
                nn.Sequential(
                    Conv(ch, 64, 3),
                    Conv(64, 64, 3),
                    nn.Conv2d(64, nc, 1),
                )
            )
            self.cv4.append(
                nn.Sequential(
                    Conv(ch, 48, 3),
                    Conv(48, 48, 3),
                    nn.Conv2d(48, nk * KPT_DIM, 1),
                )
            )

    def forward(self, xs):
        """xs: 三个尺度的特征图 [B,64,60,80], [B,128,30,40], [B,256,15,20]
        返回每个尺度的 [B, 3+15, H, W]。
        """
        outs = []
        for i, x in enumerate(xs):
            cls = self.cv3[i](x)
            kpt = self.cv4[i](x)
            outs.append(torch.cat((cls, kpt), dim=1))
        return outs


class RuneYOLOv5(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.ModuleList()
        # backbone
        self.model.append(Conv(3, 16, 3, 2))
        self.model.append(Conv(16, 32, 3, 2))
        self.model.append(C3(32, 32, 1))
        self.model.append(Conv(32, 64, 3, 2))
        self.model.append(C3(64, 64, 2))
        self.model.append(Conv(64, 128, 3, 2))
        self.model.append(C3(128, 128, 2))
        self.model.append(Conv(128, 256, 3, 2))
        self.model.append(C3(256, 256, 1))
        self.model.append(SPPF(256, 256, 128))
        # neck
        self.model.append(nn.Upsample(scale_factor=2.0, mode="nearest"))  # 10
        self.model.append(nn.Sequential())  # 11 Concat 占位
        self.model.append(C3(384, 128, 1, shortcut=False))  # 12
        self.model.append(nn.Upsample(scale_factor=2.0, mode="nearest"))  # 13
        self.model.append(nn.Sequential())  # 14 Concat 占位
        self.model.append(C3(192, 64, 1, shortcut=False))  # 15 -> P3
        self.model.append(Conv(64, 64, 3, 2))  # 16
        self.model.append(nn.Sequential())  # 17 Concat 占位
        self.model.append(C3(192, 128, 1, shortcut=False))  # 18 -> P4
        self.model.append(Conv(128, 128, 3, 2))  # 19
        self.model.append(nn.Sequential())  # 20 Concat 占位
        self.model.append(C3(384, 256, 1, shortcut=False))  # 21 -> P5
        self.detect = DetectHead((64, 128, 256), nc=NUM_CLASSES, nk=NUM_KEYPOINTS)

    def forward(self, x):
        y = self.model[0](x)
        y = self.model[1](y)
        f2 = self.model[2](y)
        y = self.model[3](f2)
        f4 = self.model[4](y)
        y = self.model[5](f4)
        f6 = self.model[6](y)
        y = self.model[7](f6)
        f8 = self.model[8](y)
        f9 = self.model[9](f8)
        f10 = self.model[10](f9)
        f11 = torch.cat((f10, f6), dim=1)
        f12 = self.model[12](f11)
        f13 = self.model[13](f12)
        f14 = torch.cat((f13, f4), dim=1)
        f15 = self.model[15](f14)  # P3/8
        f16 = self.model[16](f15)
        f17 = torch.cat((f16, f12), dim=1)
        f18 = self.model[18](f17)  # P4/16
        f19 = self.model[19](f18)
        f20 = torch.cat((f19, f9), dim=1)
        f21 = self.model[21](f20)  # P5/32
        return self.detect([f15, f18, f21])


def decode_outputs(outs, input_h=480, input_w=640):
    """把检测头原始输出 [B, 3+15, H, W] 解码成 ONNX 的 [B, 18, 6300] 形式。

    与 ONNX 图尾部的 decode 链一致（由 Constant/Slice 节点实测确定）：
      类别/关键点 conf = sigmoid(raw)
      关键点 xy = (raw*2 + 网格单元) * stride     ← 注意：无 sigmoid、无 -0.5
    通道顺序: [cls0, cls1, cls2, k0x, k0y, k0c, ..., k4x, k4y, k4c]
    """
    bs = outs[0].shape[0]
    chunks = []
    for layer, out in enumerate(outs):
        stride = STRIDES[layer]
        b, c, h, w = out.shape
        cls, kpt = out.split([NUM_CLASSES, NUM_KEYPOINTS * KPT_DIM], dim=1)
        cls = torch.sigmoid(cls)
        kpt = kpt.reshape(bs, NUM_KEYPOINTS, KPT_DIM, h, w)
        kxy, kconf = kpt[:, :, :2], kpt[:, :, 2:3]
        gy, gx = torch.meshgrid(
            torch.arange(h, dtype=torch.float32),
            torch.arange(w, dtype=torch.float32),
            indexing="ij",
        )
        kxy[:, :, 0] = (kxy[:, :, 0] * 2.0 + gx) * stride
        kxy[:, :, 1] = (kxy[:, :, 1] * 2.0 + gy) * stride
        kconf = torch.sigmoid(kconf)
        kpt = torch.cat((kxy, kconf), dim=2)  # [B,5,3,H,W]
        out = torch.cat((cls, kpt.reshape(bs, -1, h, w)), dim=1)  # [B,18,H,W]
        chunks.append(out.reshape(bs, 18, -1))
    return torch.cat(chunks, dim=2)  # [B,18,6300]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    model = RuneYOLOv5()
    model.eval()
    x = torch.randn(1, 3, 480, 640)
    with torch.no_grad():
        outs = model(x)
        for o in outs:
            print("raw head:", tuple(o.shape))
        out = decode_outputs(outs)
        print("decoded:", tuple(out.shape))
    print("params:", count_parameters(model))
