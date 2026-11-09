# -*- coding: utf-8 -*-
"""
Đánh giá độ chính xác nhận diện NHIỀU tờ / tờ bị đè trên một bộ ảnh có nhãn.

Bộ ảnh là một thư mục có labels.json dạng:
  [{"file": "a.jpg", "notes": [{"denom": 500000, "quad": [[x,y],...4 góc], "visible": 0.6}, ...]}, ...]
- "quad" (khung đầy đủ của tờ) và "visible" là tuỳ chọn. Với ẢNH CHỤP THẬT chỉ cần ghi mệnh giá:
  {"file": "IMG_01.jpg", "notes": [{"denom": 500000}, {"denom": 20000}]}
  khi đó chỉ chấm theo danh sách mệnh giá và tổng tiền, không chấm vị trí.
- tao_bo_test.py sinh sẵn bộ ảnh ghép có đủ nhãn.

Chỉ số:
  - Phát hiện: một tờ được coi là tìm thấy nếu có khung dự đoán IoU >= 0.5 với khung thật.
  - Đúng hoàn toàn: tìm thấy VÀ đúng mệnh giá.
  - Theo danh sách mệnh giá: số tờ đúng khi so tập mệnh giá dự đoán với tập thật (không xét vị trí).
  - Ảnh đúng tổng tiền: tổng dự đoán bằng đúng tổng thật.

Chạy:
  python danh_gia.py bo_test                          # cách mới (SIFT + RANSAC lặp)
  python danh_gia.py bo_test --feature orb            # cách mới dùng ORB (nhanh hơn)
  python danh_gia.py bo_test --method classic         # cách cũ để so sánh
  python danh_gia.py bo_test --save-errors loi/       # lưu ảnh các trường hợp sai
"""
import argparse
import json
import os
import sys
import time
from collections import Counter

import cv2
import numpy as np

import recognizer as R

VIS_BINS = [(0.0, 0.4, "nhìn thấy <40%"), (0.4, 0.7, "nhìn thấy 40-70%"), (0.7, 1.01, "nhìn thấy >=70%")]


def match_notes(gt, preds, scale):
    """Ghép tờ thật với dự đoán theo IoU giảm dần. Trả về {chỉ số tờ thật: dự đoán}."""
    pairs = []
    for i, g in enumerate(gt):
        q = np.array(g["quad"], np.float32) * scale
        for j, p in enumerate(preds):
            iou = R._quad_iou(q, p.quad)
            if iou >= 0.5:
                pairs.append((iou, i, j))
    out, used = {}, set()
    for iou, i, j in sorted(pairs, reverse=True):
        if i not in out and j not in used:
            out[i] = preds[j]
            used.add(j)
    return out


def draw_case(img, gt, preds, scale):
    vis = R.draw_results(img, preds)
    for g in gt:
        if "quad" in g:
            q = (np.array(g["quad"]) * scale).astype(np.int32)
            cv2.polylines(vis, [q], True, (255, 128, 0), 1)
            cv2.putText(vis, str(g["denom"]), tuple(int(v) for v in q.mean(axis=0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), 2)
    return vis


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--method", default="multi", choices=["multi", "classic", "color"])
    ap.add_argument("--feature", default=R.FEATURE_KIND, choices=["sift", "orb"])
    ap.add_argument("--save-errors", default=None, help="thư mục lưu ảnh các trường hợp sai")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    with open(os.path.join(a.folder, "labels.json"), encoding="utf-8") as f:
        labels = json.load(f)
    if a.limit:
        labels = labels[:a.limit]
    rec = R.BanknoteRecognizer(feature_kind=a.feature)
    if a.save_errors:
        os.makedirs(a.save_errors, exist_ok=True)

    n_gt = n_found = n_exact = n_pred = n_pred_ok = n_uncertain = 0
    n_bag_ok = n_img_total_ok = n_img_all_ok = 0
    by_vis = {b[2]: [0, 0] for b in VIS_BINS}
    by_layout = {}
    confusion = Counter()
    times = []
    for item in labels:
        img = R.imread_unicode(os.path.join(a.folder, item["file"]))
        if img is None:
            print("Không đọc được", item["file"])
            continue
        t0 = time.perf_counter()
        small, res = rec.recognize(img, use_features=a.method != "color",
                                   method="classic" if a.method == "color" else a.method)
        times.append(time.perf_counter() - t0)
        scale = small.shape[1] / img.shape[1]
        gt = item["notes"]
        preds = [r for r in res if r.denom]
        n_uncertain += len(res) - len(preds)

        # chấm theo danh sách mệnh giá (không xét vị trí)
        gt_bag, pred_bag = Counter(g["denom"] for g in gt), Counter(p.denom for p in preds)
        bag_ok = sum((gt_bag & pred_bag).values())
        n_bag_ok += bag_ok
        n_pred += len(preds)
        total_ok = sum(gt_bag.elements()) == sum(pred_bag.elements())
        n_img_total_ok += total_ok

        # chấm theo vị trí
        exact_img = 0
        if all("quad" in g for g in gt):
            m = match_notes(gt, preds, scale)
            for i, g in enumerate(gt):
                p = m.get(i)
                ok = p is not None and p.denom == g["denom"]
                n_found += p is not None
                exact_img += ok
                if p is not None and not ok:
                    confusion[(g["denom"], p.denom)] += 1
                if p is None:
                    confusion[(g["denom"], "bỏ sót")] += 1
                v = g.get("visible", 1.0)
                for lo, hi, name in VIS_BINS:
                    if lo <= v < hi:
                        by_vis[name][0] += ok
                        by_vis[name][1] += 1
            n_exact += exact_img
        else:
            exact_img = bag_ok
        n_pred_ok += exact_img          # mỗi dự đoán ghép với tối đa một tờ thật
        n_gt += len(gt)
        all_ok = exact_img == len(gt) and len(preds) == len(gt)
        n_img_all_ok += all_ok
        lay = by_layout.setdefault(item.get("layout", "-"), [0, 0])
        lay[0] += all_ok
        lay[1] += 1

        if a.save_errors and not all_ok:
            cv2.imwrite(os.path.join(a.save_errors, item["file"]), draw_case(small, gt, preds, scale))

    n_img = len(times)
    pct = lambda x, y: f"{x}/{y} = {x / y * 100:5.1f}%" if y else "-"
    print(f"\n=== {a.folder} | cách: {a.method}" + (f" ({a.feature})" if a.method == "multi" else "") + " ===")
    print(f"Ảnh: {n_img}, tờ thật: {n_gt}, thời gian trung bình {np.mean(times) * 1000:.0f} ms/ảnh")
    if n_found or n_exact:
        print(f"Phát hiện (IoU>=0.5)         : {pct(n_found, n_gt)}")
        print(f"Đúng hoàn toàn (vị trí+mệnh giá): {pct(n_exact, n_gt)}")
    print(f"Đúng theo danh sách mệnh giá : {pct(n_bag_ok, n_gt)}")
    print(f"Độ chính xác dự đoán (precision): {pct(n_pred_ok, n_pred)}   | 'không chắc chắn': {n_uncertain}")
    print(f"Ảnh đúng tổng tiền           : {pct(n_img_total_ok, n_img)}")
    print(f"Ảnh đúng TẤT CẢ các tờ       : {pct(n_img_all_ok, n_img)}")
    if any(v[1] for v in by_vis.values()):
        print("Đúng hoàn toàn theo phần nhìn thấy:")
        for name, (ok, n) in by_vis.items():
            print(f"   {name:18s}: {pct(ok, n)}")
    if len(by_layout) > 1:
        print("Ảnh đúng tất cả theo kiểu bố trí:")
        for name, (ok, n) in sorted(by_layout.items()):
            print(f"   {name:6s}: {pct(ok, n)}")
    if confusion:
        print("Lỗi thường gặp (thật -> dự đoán):",
              ", ".join(f"{k[0]}->{k[1]}: {v}" for k, v in confusion.most_common(8)))


if __name__ == "__main__":
    main()
