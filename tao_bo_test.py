# -*- coding: utf-8 -*-
"""
Tạo bộ ảnh test TIỀN BỊ ĐÈ có nhãn chuẩn (ground truth) để đo độ chính xác.

Mỗi ảnh ghép 2-6 tờ từ ảnh mẫu lên nền ngẫu nhiên, theo 3 kiểu bố trí:
  - "de"   : các tờ đè lên nhau ngẫu nhiên (xoay, dịch, phối cảnh)
  - "quat" : xòe như quạt, các tờ che nhau rất nhiều (giống cầm tiền trên tay)
  - "sat"  : các tờ nằm sát / chạm nhau, đè ít
rồi làm xấu ảnh: bóng đổ, ánh sáng không đều, ám màu, nhiễu, mờ, nén JPEG, đổi độ phân giải.

Nhãn lưu ở <thư mục>/labels.json: với mỗi tờ có mệnh giá, 4 góc của khung đầy đủ
(kể cả phần bị che) và tỉ lệ diện tích còn nhìn thấy.

LƯU Ý: tờ tiền được lấy từ chính ảnh mẫu nên kết quả trên bộ này LẠC QUAN hơn ảnh chụp
thật. Hãy bổ sung ảnh chụp thật để đánh giá cuối cùng (xem danh_gia.py).

Chạy:  python tao_bo_test.py --out bo_test --n 150 --seed 1
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

import recognizer as R

SRC_W, SRC_H = 1000, 450
CANVAS_W, CANVAS_H = 1800, 1350


def load_sources(folder):
    srcs = []
    for rel, path, denom, _ in R._scan_template_files(folder):
        img = R.imread_unicode(path)
        if img is not None:
            srcs.append((denom, rel, R.extract_main_note(img, (SRC_W, SRC_H))))
    return srcs


def random_background(rng, w, h):
    kind = rng.choice(["tron", "chuyen_mau", "van_go", "nhieu"])
    base = rng.integers(30, 240, 3).astype(np.float32)
    if kind == "tron":
        bg = np.ones((h, w, 3), np.float32) * base
    elif kind == "chuyen_mau":
        other = rng.integers(30, 240, 3).astype(np.float32)
        t = np.linspace(0, 1, w, dtype=np.float32)[None, :, None]
        bg = base * (1 - t) + other * t + np.zeros((h, 1, 1), np.float32)
    elif kind == "van_go":
        y = np.arange(h, dtype=np.float32)[:, None]
        x = np.arange(w, dtype=np.float32)[None, :]
        stripes = np.sin(y / rng.uniform(6, 20) + 3 * np.sin(x / rng.uniform(80, 200)))
        bg = base * (0.85 + 0.15 * stripes)[..., None]
    else:
        n = rng.normal(0, 1, (h // 16 + 1, w // 16 + 1, 3)).astype(np.float32)
        n = cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC)
        bg = base + 25 * n
    return np.clip(bg, 0, 255)


def note_quad(rng, center, width, angle_deg, persp=0.03):
    h = width * SRC_H / SRC_W
    c, s = np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))
    base = np.array([[-width / 2, -h / 2], [width / 2, -h / 2], [width / 2, h / 2], [-width / 2, h / 2]])
    quad = base @ np.array([[c, s], [-s, c]]) + center
    return (quad + rng.normal(0, persp * width, (4, 2))).astype(np.float32)


def layout(rng, n, w, h):
    """Trả về danh sách (kiểu bố trí, [quad...]) theo thứ tự từ dưới lên trên."""
    kind = rng.choice(["de", "quat", "sat"], p=[0.6, 0.25, 0.15])
    nw = rng.uniform(0.30, 0.46) * w                     # chiều rộng tờ tiền trên ảnh
    quads = []
    if kind == "quat":
        pivot = np.array([rng.uniform(0.35, 0.65) * w, rng.uniform(0.55, 0.75) * h])
        a0, step = rng.uniform(0, 360), rng.uniform(9, 22) * rng.choice([-1, 1])
        for i in range(n):
            a = a0 + i * step
            d = np.array([np.cos(np.radians(a)), np.sin(np.radians(a))])
            quads.append(note_quad(rng, pivot + d * nw * 0.42, nw, a))
    elif kind == "sat":
        a0 = rng.uniform(-25, 25)
        nh = nw * SRC_H / SRC_W
        start = np.array([w / 2, h / 2]) - np.array([0, (n - 1) * nh * 0.5])
        for i in range(n):
            c = start + np.array([rng.normal(0, nw * 0.04), i * nh * rng.uniform(0.85, 1.0)])
            quads.append(note_quad(rng, c, nw, a0 + rng.normal(0, 4)))
    else:
        centers = [np.array([rng.uniform(0.3, 0.7) * w, rng.uniform(0.3, 0.7) * h])]
        for _ in range(n - 1):
            ref = centers[rng.integers(len(centers))]
            phi = rng.uniform(0, 2 * np.pi)
            c = ref + rng.uniform(0.25, 0.7) * nw * np.array([np.cos(phi), np.sin(phi)])
            centers.append(np.clip(c, [0.15 * w, 0.15 * h], [0.85 * w, 0.85 * h]))
        quads = [note_quad(rng, c, nw * rng.uniform(0.9, 1.1), rng.uniform(0, 360)) for c in centers]
    return kind, quads


def render(rng, srcs, n):
    w, h = CANVAS_W, CANVAS_H
    kind, quads = layout(rng, n, w, h)
    canvas = random_background(rng, w, h)
    owner = np.full((h, w), -1, np.int32)
    picks = [srcs[i] for i in rng.integers(len(srcs), size=n)]
    src_corners = R._rect_corners(SRC_W, SRC_H)
    for i, ((denom, rel, note), q) in enumerate(zip(picks, quads)):
        M = cv2.getPerspectiveTransform(src_corners, q)
        warped = cv2.warpPerspective(note, M, (w, h)).astype(np.float32)
        mask = cv2.warpPerspective(np.full((SRC_H, SRC_W), 255, np.uint8), M, (w, h))
        # bóng đổ mềm dưới tờ tiền
        shadow = cv2.GaussianBlur(mask, (0, 0), 9).astype(np.float32) / 255
        shadow = np.roll(shadow, (int(rng.integers(3, 10)), int(rng.integers(3, 10))), axis=(0, 1))
        canvas *= (1 - 0.35 * shadow)[..., None]
        alpha = cv2.GaussianBlur(mask, (3, 3), 0).astype(np.float32)[..., None] / 255
        canvas = canvas * (1 - alpha) + warped * alpha
        owner[mask > 127] = i

    # làm xấu ảnh: ánh sáng không đều, ám màu, nhiễu, mờ
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    gx, gy = rng.uniform(-0.35, 0.35, 2)
    light = 1 + gx * (xx / w - 0.5) + gy * (yy / h - 0.5)
    light *= rng.uniform(0.55, 1.15)
    canvas = canvas * light[..., None] * rng.uniform(0.82, 1.18, 3)
    canvas += rng.normal(0, rng.uniform(0, 7), canvas.shape)
    img = np.clip(canvas, 0, 255).astype(np.uint8)
    k = int(rng.choice([0, 3, 5, 7]))
    if k:
        img = cv2.GaussianBlur(img, (k, k), 0)

    notes = []
    for i, ((denom, rel, _), q) in enumerate(zip(picks, quads)):
        full = cv2.contourArea(q.reshape(-1, 1, 2))
        notes.append({"denom": int(denom), "src": rel, "quad": q.tolist(),
                      "visible": round(float(np.count_nonzero(owner == i)) / full, 3)})
    return kind, img, notes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="bo_test")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--templates", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates"))
    ap.add_argument("--min-visible", type=float, default=0.2,
                    help="bỏ cảnh có tờ còn nhìn thấy ít hơn tỉ lệ này")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    rng = np.random.default_rng(a.seed)
    srcs = load_sources(a.templates)
    os.makedirs(a.out, exist_ok=True)
    labels, made = [], 0
    while made < a.n:
        n = int(rng.choice([2, 3, 4, 5, 6], p=[0.25, 0.3, 0.2, 0.15, 0.1]))
        kind, img, notes = render(rng, srcs, n)
        if min(x["visible"] for x in notes) < a.min_visible:
            continue
        s = rng.uniform(0.5, 1.0)                     # độ phân giải cuối cùng
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        for x in notes:
            x["quad"] = (np.array(x["quad"]) * s).round(1).tolist()
        name = f"{made:04d}_{kind}_{n}to.jpg"
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(45, 92))])
        buf.tofile(os.path.join(a.out, name))
        labels.append({"file": name, "layout": kind, "notes": notes})
        made += 1
    with open(os.path.join(a.out, "labels.json"), "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=1)
    vis = [x["visible"] for l in labels for x in l["notes"]]
    print(f"Đã tạo {made} ảnh, {len(vis)} tờ vào {a.out}/ "
          f"(phần nhìn thấy: nhỏ nhất {min(vis):.0%}, trung vị {np.median(vis):.0%})")


if __name__ == "__main__":
    main()
