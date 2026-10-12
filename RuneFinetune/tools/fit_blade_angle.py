#!/usr/bin/env python3
"""Blade-angle convention fitting for Daedalus rune GT projection.

Part A: fine-scan rune0 C constant (step 1 deg) on frames with r0 lit blades
        {0,3}, scoring red pixels inside projected module circles.
Part B: scan rune1 (direction=-1) formula variants on both-lit frames,
        excluding red pixels inside rune0's (known) lit module circles.
Part C: render verification overlays with best-fit parameters.
"""
import sys, os, json, glob, math
from PIL import Image, ImageDraw, ImageChops

FX = FY = 1303.67532368147
CX, CY = 720.0, 540.0
W, H = 1440, 1080
MODULE_R = 0.15
MODULE_DIST = 0.70
BOARD_R = 0.78
CROP = (500, 260, 900, 660)  # (x0, y0, x1, y1) region containing both runes

def quat_to_R(q):
    w, x, y, z = [float(v) for v in q]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return ((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
            (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
            (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)))

def project(p, R, t):
    dx, dy, dz = p[0] - t[0], p[1] - t[1], p[2] - t[2]
    xc = R[0][0] * dx + R[1][0] * dy + R[2][0] * dz
    yc = R[0][1] * dx + R[1][1] * dy + R[2][1] * dz
    zc = R[0][2] * dx + R[1][2] * dy + R[2][2] * dz
    if xc < 0.01:
        return None
    return (FX * (-yc) / xc + CX, FY * (-zc) / xc + CY)

def module_pts(center, phi, R, t, n=20):
    d = (0.0, math.cos(phi), math.sin(phi))
    orth = (0.0, -math.sin(phi), math.cos(phi))
    mc = (center[0] + MODULE_DIST * d[0],
          center[1] + MODULE_DIST * d[1],
          center[2] + MODULE_DIST * d[2])
    pts = []
    for i in range(n):
        th = 2 * math.pi * i / n
        p = (mc[0] + MODULE_R * (math.cos(th) * orth[0] + math.sin(th) * d[0]),
             mc[1] + MODULE_R * (math.cos(th) * orth[1] + math.sin(th) * d[1]),
             mc[2] + MODULE_R * (math.cos(th) * orth[2] + math.sin(th) * d[2]))
        uv = project(p, R, t)
        if uv is None:
            return None
        pts.append(uv)
    return pts

def disk_pts(center, radius, R, t, n=28):
    pts = []
    for i in range(n):
        th = 2 * math.pi * i / n
        uv = project((center[0], center[1] + radius * math.cos(th),
                      center[2] + radius * math.sin(th)), R, t)
        if uv is None:
            return None
        pts.append(uv)
    return pts

def red_mask(img):
    r, g, b = img.split()
    g15 = g.point(lambda v: min(255, int(v * 1.5)))
    b15 = b.point(lambda v: min(255, int(v * 1.5)))
    m = ImageChops.multiply(r.point(lambda v: 255 if v > 100 else 0),
                            ImageChops.subtract(r, g15).point(lambda v: 255 if v > 0 else 0))
    m = ImageChops.multiply(m, ImageChops.subtract(r, b15).point(lambda v: 255 if v > 0 else 0))
    return m

def poly_count(mask, pts, crop=None):
    box = crop if crop else (0, 0, W, H)
    ox, oy = box[0], box[1]
    poly = Image.new('L', (box[2] - box[0], box[3] - box[1]), 0)
    d = ImageDraw.Draw(poly)
    d.polygon([(p[0] - ox, p[1] - oy) for p in pts], fill=255)
    src = mask.crop(box) if crop else mask
    inter = ImageChops.multiply(src, poly)
    return inter.histogram()[255]

class Frame:
    __slots__ = ('seq', 'j', 'img', 'mask', 'ddir')
    def __init__(self, j, ddir):
        self.seq = int(j['frame_seq'])
        self.j = j
        self.img = None
        self.mask = None
        self.ddir = ddir
    def get_img(self):
        if self.img is None:
            self.img = Image.open(os.path.join(self.ddir, self.j['image']['file'])).convert('RGB')
        return self.img
    def get_mask(self):
        if self.mask is None:
            self.mask = red_mask(self.get_img())
        return self.mask
    def cam(self):
        e = self.j['exposure']
        return quat_to_R(e['camera_quat_world_wxyz']), e['camera_position_world']

def load_frames(ddir):
    out = []
    for p in sorted(glob.glob(os.path.join(ddir, 'frame_*.json'))):
        out.append(Frame(json.load(open(p)), ddir))
    return out

def rune_by_team(f, team):
    for r in f.j['runes']:
        if r['team'] == team:
            return r
    return None

def main():
    ddir = sys.argv[1] if len(sys.argv) > 1 else '/tmp/rune_seq3'
    frames = load_frames(ddir)
    print(f'loaded {len(frames)} frames')

    # ---- diagnostic: is rune0 board red when nothing is lit? ----
    dark = [f for f in frames if sum(rune_by_team(f, 0)['target_activations']) == 0]
    red_in_board = []
    for f in dark[:20]:
        R, t = f.cam()
        r0 = rune_by_team(f, 0)
        pts = disk_pts(r0['center'], BOARD_R, R, t)
        if pts:
            red_in_board.append(poly_count(f.get_mask(), pts))
    print(f'diag: red px inside r0 board disk (r0 unlit frames, n={len(red_in_board)}): '
          f'max={max(red_in_board) if red_in_board else None}')

    # ---- Part A: fine scan rune0 ----
    fa = [f for f in frames if rune_by_team(f, 0)['target_activations'] == [1, 0, 0, 1, 0]]
    print(f'Part A frames: {len(fa)}')
    results = []
    for form, crange in (('A: C-a+k*72', range(275, 296)), ('B: C-a-k*72', range(60, 81))):
        for C in crange:
            tot = 0
            for f in fa:
                R, t = f.cam()
                r0 = rune_by_team(f, 0)
                a = r0['current_angle']
                mask = f.get_mask()
                for k in (0, 3):
                    if form.startswith('A'):
                        phi = math.radians(C) - a + k * 2 * math.pi / 5
                    else:
                        phi = math.radians(C) - a - k * 2 * math.pi / 5
                    pts = module_pts(r0['center'], phi, R, t)
                    if pts:
                        tot += poly_count(mask, pts)
            results.append((form, C, tot))
    results.sort(key=lambda x: -x[2])
    print('Part A top 5:')
    for form, C, tot in results[:5]:
        print(f'  {form}  C={C}  red={tot}')
    bestA = results[0]

    # ---- Part B: rune1 variants ----
    fb = [f for f in frames
          if sum(rune_by_team(f, 0)['target_activations']) > 0
          and sum(rune_by_team(f, 1)['target_activations']) > 0]
    print(f'Part B frames: {len(fb)}')

    C0_form, C0 = bestA[0][0], bestA[1]
    def r0_excl(f):
        R, t = f.cam()
        r0 = rune_by_team(f, 0)
        a = r0['current_angle']
        excl = Image.new('L', (CROP[2] - CROP[0], CROP[3] - CROP[1]), 0)
        d = ImageDraw.Draw(excl)
        for k in range(5):
            if r0['target_activations'][k]:
                if C0_form.startswith('A'):
                    phi = math.radians(C0) - a + k * 2 * math.pi / 5
                else:
                    phi = math.radians(C0) - a - k * 2 * math.pi / 5
                pts = module_pts(r0['center'], phi, R, t)
                if pts:
                    d.polygon([(p[0] - CROP[0], p[1] - CROP[1]) for p in pts], fill=255)
        return excl

    variants = {
        'V1 C-a-k72': lambda C, a, k: math.radians(C) - a - k * 2 * math.pi / 5,
        'V2 C+a+k72': lambda C, a, k: math.radians(C) + a + k * 2 * math.pi / 5,
        'V3 C-a+k72': lambda C, a, k: math.radians(C) - a + k * 2 * math.pi / 5,
        'V4 C+a-k72': lambda C, a, k: math.radians(C) + a - k * 2 * math.pi / 5,
    }
    excl_cache = {f.seq: r0_excl(f) for f in fb}
    print('Part B coarse scan (step 5 deg)...')
    coarse = {name: [] for name in variants}
    for C in range(0, 360, 5):
        for name, fn in variants.items():
            tot = 0
            for f in fb:
                R, t = f.cam()
                r1 = rune_by_team(f, 1)
                a = r1['current_angle']
                mask = f.get_mask().crop(CROP)
                excl = excl_cache[f.seq]
                for k in range(5):
                    if r1['target_activations'][k]:
                        pts = module_pts(r1['center'], fn(C, a, k), R, t)
                        if pts:
                            poly = Image.new('L', (CROP[2] - CROP[0], CROP[3] - CROP[1]), 0)
                            d = ImageDraw.Draw(poly)
                            d.polygon([(p[0] - CROP[0], p[1] - CROP[1]) for p in pts], fill=255)
                            inter = ImageChops.subtract(ImageChops.multiply(mask, poly), excl)
                            tot += inter.histogram()[255]
            coarse[name].append((C, tot))
    for name in variants:
        coarse[name].sort(key=lambda x: -x[1])
        print(f'  {name} best coarse: C={coarse[name][0][0]} red={coarse[name][0][1]}')
    print('Part B fine scan (step 1 deg around best)...')
    for name in variants:
        bestC = coarse[name][0][0]
        fine = []
        for C in range(bestC - 6, bestC + 7):
            tot = 0
            fn = variants[name]
            for f in fb:
                R, t = f.cam()
                r1 = rune_by_team(f, 1)
                a = r1['current_angle']
                mask = f.get_mask().crop(CROP)
                excl = excl_cache[f.seq]
                for k in range(5):
                    if r1['target_activations'][k]:
                        pts = module_pts(r1['center'], fn(C, a, k), R, t)
                        if pts:
                            poly = Image.new('L', (CROP[2] - CROP[0], CROP[3] - CROP[1]), 0)
                            d = ImageDraw.Draw(poly)
                            d.polygon([(p[0] - CROP[0], p[1] - CROP[1]) for p in pts], fill=255)
                            inter = ImageChops.subtract(ImageChops.multiply(mask, poly), excl)
                            tot += inter.histogram()[255]
            fine.append((C, tot))
        fine.sort(key=lambda x: -x[1])
        print(f'  {name} best fine: C={fine[0][0]} red={fine[0][1]}')
        coarse[name] = [(fine[0][0], fine[0][1])] + coarse[name]

    # ---- Part C: render overlays ----
    render_seqs = []
    for want in ([1, 0, 0, 1, 0], [0, 1, 0, 0, 1]):
        for f in frames:
            if rune_by_team(f, 0)['target_activations'] == want:
                render_seqs.append(f.seq)
                break
    print(f'render frames: {render_seqs}')
    for seq in render_seqs:
        f = next(x for x in frames if x.seq == seq)
        img = f.get_img().copy()
        d = ImageDraw.Draw(img, 'RGBA')
        R, t = f.cam()
        for team, col, litcol, lab in ((0, (0, 255, 0, 255), (255, 0, 0, 255), 'R'),
                                      (1, (0, 255, 255, 255), (255, 0, 255, 255), 'B')):
            r = rune_by_team(f, team)
            a = r['current_angle']
            for k in range(5):
                if team == 0:
                    phi = (math.radians(C0) - a + k * 2 * math.pi / 5
                           if C0_form.startswith('A')
                           else math.radians(C0) - a - k * 2 * math.pi / 5)
                else:
                    name = max(variants, key=lambda n: coarse[n][0][1])
                    phi = variants[name](coarse[name][0][0], a, k)
                pts = module_pts(r['center'], phi, R, t)
                if pts:
                    c = litcol if r['target_activations'][k] else col
                    d.polygon([(p[0], p[1]) for p in pts], outline=c)
                    if r['target_activations'][k]:
                        d.text((pts[0][0] + 4, pts[0][1] + 4), f'{lab}{k}', fill=c)
            uv = project(r['center'], R, t)
            if uv:
                d.ellipse((uv[0] - 4, uv[1] - 4, uv[0] + 4, uv[1] + 4), outline=col)
        out = f'/tmp/verify_blades_{seq}.jpg'
        img.save(out, quality=90)
        print(f'saved {out}')

if __name__ == '__main__':
    main()
