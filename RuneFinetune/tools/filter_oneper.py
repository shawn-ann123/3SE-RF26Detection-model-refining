"""把 labels.jsonl 过滤为 labels_oneper.jsonl：每帧每类最多保留一个对象。

模型每个锚点单元只能表示一组关键点（按类别分派到三个尺度），
同帧同类多个对象共享 R 中心会互相覆盖，训练前必须过滤。
保留每类第一个出现的目标（labels.jsonl 内顺序）。

用法：
  .venv-xpu/bin/python tools/filter_oneper.py [labels.jsonl 路径] [输出路径]
默认：data/runeset/labels.jsonl -> data/runeset/labels_oneper.jsonl
"""

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "runeset" / "labels.jsonl"
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_name("labels_oneper.jsonl")
    rows = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
    per_frame = Counter()
    per_cls = Counter()
    out = []
    for r in rows:
        kept, seen = [], set()
        for obj in r["objects"]:
            c = obj["class_id"]
            if c in seen:
                continue
            seen.add(c)
            kept.append(obj)
            per_cls[c] += 1
        per_frame[len(kept)] += 1
        out.append({"file": r["file"], "workflow": r.get("workflow", ""), "objects": kept})
    with open(dst, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{src} -> {dst}")
    print(f"records: {len(out)}   per-frame obj counts: {dict(sorted(per_frame.items()))}")
    print(f"per-class totals: {dict(sorted(per_cls.items()))}")


if __name__ == "__main__":
    main()
