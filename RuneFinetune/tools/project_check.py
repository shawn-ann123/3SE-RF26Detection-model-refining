#!/usr/bin/env python3
"""将 RP-26Rune 模型点按模拟器真值位姿投影到抓取帧上，验证标注几何约定。

用法: python3 project_check.py <truth.txt> <frame.ppm> [--out out.jpg]
未知约定（法向符号、叶片零角、模块方向）作为参数扫一遍，输出多张对比图。
"""
import math
import re
import sys
import argparse

import numpy as np
from PIL import Image, ImageDraw

# 相机标定 (camera-calibration.json)
FX = FY = 1303.67532368147
CX, CY = 720.0, 540.0

# 模型点（points_world.cpp，符心为原点，1号位叶片指向 -y）
MODULE_CENTER_R = 0.70          # 靶心 (0,-0.7,0)
MODULE_CIRCLE_R = 0.15          # 未激活/小符装甲模块圆半径
ARM_HALF_W = 0.030              # 灯臂半宽
ARM_Y_INNER = -0.173940         # 灯臂内端
ARM_Y_OUTER = -0.503940         # 灯臂外端


def parse_truth(path):
    text = open(path).read()
    runes = []
    for m in re.finditer(
        r"-- rune\[(\d+)\] --.*?r_center_odom=\[([-\d.]+), ([-\d.]+), ([-\d.]+)\].*?"
        r"current_angle=([-\d.]+).*?direction=(-?\d+).*?"
        r"target_activations=\[(\d),(\d),(\d),(\d),(\d)\]",
        text, re.S):
        runes.append(dict(
            idx=int(m.group(1)),
            center=np.array([float(m.group(2)), float(m.group(3)), float(m.group(4))]),
            angle=float(m.group(5)),
            direction=int(m.group(6)),
            activations=[int(m.group(i)) for i in range(7, 12)],
        ))
    cam = dict(
        t=np.array([float(x) for x in re.search(
            r"camera_position_world=\[([-\d.]+), ([-\d.]+), ([-\d.]+)\]", text).groups()]),
        q=np.array([float(x) for x in re.search(
            r"camera_quat_world_wxyz=\[([-\d.]+), ([-\d.]+), ([-\d.]+), ([-\d.]+)\]",
            text).groups()]),  # wxyz
    )
    return runes, cam


def quat_to_mat(q_wxyz):
    """四元数 wxyz -> 旋转矩阵 (该旋转把本体系坐标转到世界系)。"""
    x, y, z, w = q_wxyz
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def project(X_world, t, R):
    """世界点 -> 像素。R: 相机在世界系的旋转矩阵。"""
    Xc = R.T @ (X_world - t)
    u = FX * Xc[0] / Xc[2] + CX
    v = FY * Xc[1] / Xc[2] + CY
    return np.array([u, v])


def rune_model_points(center, angle0, normal_sign, module_sign, blade_dir0,
                      module_center_r=MODULE_CENTER_R):
    """生成 5 片叶的模型点(世界系)。

    angle0: 0 号叶片的方向角
    normal_sign: 盘面法向 = normal_sign * x_odom
    module_sign: 模块在叶片方向上位于 +R 还是 -R（对应模型 y 轴方向）
    blade_dir0: 叶片零角的基向量 (yz 平面内, 单位向量)
    模型系 (x_m, y_m): x_m 沿盘面横向, y_m 沿叶片方向(1号位为 -y_m)
    """
    n = normal_sign * np.array([1.0, 0.0, 0.0])
    # 盘面内基: d(phi) = cos(phi)*u + sin(phi)*v, u x v = n
    u = blade_dir0
    v = np.cross(n, u)
    blade_dirs = []
    module_circles = []  # 每个叶片: 模块圆 3D 点列
    arm_rects = []       # 每个叶片: 灯臂矩形角点
    for k in range(5):
        phi = angle0 + k * 2*math.pi/5
        d = math.cos(phi)*u + math.sin(phi)*v
        lateral = np.cross(d, n)  # 模型 +x_m 方向
        blade_dirs.append(d)
        # 模块圆心: 模型 (0, -0.7, 0) -> y_m=-0.7 沿 -d? 模型 1 号位叶片指向 -y_m,
        # 模块在 -y_m 方向 -> 世界: center + 0.7 * d * module_sign_effective
        mc = center + module_center_r * d * module_sign
        circle = []
        for i in range(16):
            a = 2*math.pi*i/16
            # 圆上任一点: 模型 (r sin a, -0.7 + r cos a, 0) 在盘面内
            p = mc + MODULE_CIRCLE_R*math.sin(a)*lateral + MODULE_CIRCLE_R*math.cos(a)*(-d)*module_sign
            circle.append(p)
        module_circles.append(circle)
        # 灯臂矩形: 模型 x=±0.03, y∈[-0.50394,-0.17394]
        corners = []
        for xm in (-ARM_HALF_W, ARM_HALF_W):
            for ym in (ARM_Y_INNER, ARM_Y_OUTER):
                corners.append(center + xm*lateral + ym*d*module_sign)
        arm_rects.append(corners)
    return blade_dirs, module_circles, arm_rects, n


def draw_overlay(img, runes, cam, variant, out_path):
    draw = ImageDraw.Draw(img)
    t = cam['t']
    R = quat_to_mat(cam['q'])
    for rune in runes:
        center, angle = rune['center'], rune['angle']
        blade_dirs, circles, rects, n = rune_model_points(
            center, angle, variant['normal_sign'], variant['module_sign'],
            variant['blade_dir0'])
        # 盘心 (符心 / R 标)
        p0 = project(center, t, R)
        # 叶片轴 + 模块 + 灯臂
        for k in range(5):
            col = (255, 128, 0) if rune['activations'][k] else (140, 140, 140)
            tip = center + blade_dirs[k] * 0.85 * variant['module_sign']
            pa = project(center, t, R)
            pb = project(tip, t, R)
            draw.line([tuple(pa), tuple(pb)], fill=col, width=2)
            for p in circles[k]:
                pu = project(p, t, R)
                draw.ellipse([pu[0]-1.5, pu[1]-1.5, pu[0]+1.5, pu[1]+1.5], fill=col)
            pts = [tuple(project(c, t, R)) for c in rects[k]]
            for i in range(4):
                draw.line([pts[i], pts[(i+1) % 4]], fill=col, width=2)
        draw.ellipse([p0[0]-4, p0[1]-4, p0[0]+4, p0[1]+4], outline=(0, 255, 0), width=2)
        draw.text((p0[0]+6, p0[1]-8), f"R{rune['idx']}", fill=(0, 255, 0))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("truth")
    ap.add_argument("frame")
    ap.add_argument("--out", default="overlay_check")
    args = ap.parse_args()

    runes, cam = parse_truth(args.truth)
    base = Image.open(args.frame).convert("RGB")

    # 先单独投影两个符心，确认相机位姿与标定
    t, R = cam['t'], quat_to_mat(cam['q'])
    for r in runes:
        print(f"rune{r['idx']} center -> pixel {project(r['center'], t, R)}")

    # 扫参数组合
    variants = []
    for normal_sign in (+1.0, -1.0):
        for module_sign in (+1.0, -1.0):
            for bd in (np.array([0., 1., 0.]), np.array([0., 0., 1.]),
                       np.array([0., -1., 0.]), np.array([0., 0., -1.])):
                variants.append(dict(normal_sign=normal_sign, module_sign=module_sign,
                                     blade_dir0=bd))
    # 每个变体: 符心在画面中心附近的那张单独保存
    for vi, v in enumerate(variants):
        img = base.copy()
        draw_overlay(img, runes, cam, v, None)
        w, h = img.size
        # 裁剪: 以投影符心为中心 420x420
        p0 = project(runes[0]['center'], t, R)
        box = (int(p0[0]-210), int(p0[1]-210), int(p0[0]+210), int(p0[1]+210))
        crop = img.crop(box).resize((840, 840), Image.LANCZOS)
        crop.save(f"{args.out}_{vi:02d}.jpg", quality=92)
        print(f"saved {args.out}_{vi:02d}.jpg  ns={v['normal_sign']:+} ms={v['module_sign']:+} "
              f"bd={v['blade_dir0'].tolist()}")
    # 全图叠加第 0 号变体
    img = base.copy()
    draw_overlay(img, runes, cam, variants[0], None)
    img.save(f"{args.out}_full.jpg", quality=92)


if __name__ == "__main__":
    main()
