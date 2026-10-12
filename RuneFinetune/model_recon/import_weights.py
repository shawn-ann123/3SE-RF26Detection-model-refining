"""把 model-0624.onnx 的权重导入 PyTorch 重建模型，并对比两者输出。

对应关系：
  ONNX /model.i/...  <->  PyTorch model.model[i]...
  头部 /model.22/cv3.s/cv3.s.{0,1,2}  <->  detect.cv3[s][{0,1,2}]
       /model.22/cv4.s/cv4.s.{0,1,2}  <->  detect.cv4[s][{0,1,2}]
"""

import sys

import numpy as np
import onnx
import torch

sys.path.insert(0, "model_recon")
from yolov5_rune import RuneYOLOv5, decode_outputs  # noqa: E402

ONNX_PATH = "weights/model-0624.onnx"


def load_onnx_tensors(path):
    model = onnx.load(path)
    tensors = {}
    for init in model.graph.initializer:
        arr = onnx.numpy_helper.to_array(init)
        tensors[init.name] = arr
    return tensors


def assign(module, prefix, tensors, verbose=False):
    """按名字把 ONNX 张量塞进 torch 模块。
    ONNX initializer 命名即 torch 风格点号名：{prefix}{param_name}。
    """
    for name, param in module.named_parameters():
        onnx_key = f"{prefix}{name}"
        if onnx_key not in tensors:
            raise KeyError(f"missing ONNX tensor for {name} (looked for {onnx_key})")
        data = tensors[onnx_key]
        if param.shape != tuple(data.shape):
            raise ValueError(f"shape mismatch {name}: torch {tuple(param.shape)} vs onnx {data.shape}")
        param.data.copy_(torch.from_numpy(np.array(data, copy=True)))
        if verbose:
            print(f"  {name} <- {onnx_key} {data.shape}")


def import_weights(model, onnx_path=ONNX_PATH):
    tensors = load_onnx_tensors(onnx_path)
    for i, layer in enumerate(model.model):
        try:
            assign(layer, f"model.{i}.", tensors)
        except KeyError:
            pass  # Concat 占位层无参数
    for s in range(3):
        for j, sub in enumerate(model.detect.cv3[s]):
            assign(sub, f"model.22.cv3.{s}.{j}.", tensors)
        for j, sub in enumerate(model.detect.cv4[s]):
            assign(sub, f"model.22.cv4.{s}.{j}.", tensors)
    return model


def onnx_forward(onnx_path, image_np):
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"images": image_np})[0]
    return out


def main():
    torch.manual_seed(0)
    model = import_weights(RuneYOLOv5())
    model.eval()

    image_np = np.random.RandomState(0).rand(1, 3, 480, 640).astype(np.float32)

    onnx_out = onnx_forward(ONNX_PATH, image_np)
    print("onnx output:", onnx_out.shape)

    with torch.no_grad():
        x = torch.from_numpy(image_np)
        outs = model(x)
        torch_out = decode_outputs(outs).numpy()

    print("torch output:", torch_out.shape)
    diff = np.abs(onnx_out - torch_out)
    print(f"max abs diff: {diff.max():.6e}")
    print(f"mean abs diff: {diff.mean():.6e}")

    # 按尺度分段比较
    for name, lo, hi in [("P3", 0, 4800), ("P4", 4800, 6000), ("P5", 6000, 6300)]:
        d = diff[0, :, lo:hi]
        print(f"  {name}: max {d.max():.6e} mean {d.mean():.6e}")

    if diff.max() < 1e-2:
        print("OK: 重建模型与 ONNX 输出一致")
    else:
        print("MISMATCH: 输出不一致，需要排查")
    return diff.max()


if __name__ == "__main__":
    main()
