# -*- coding: utf-8 -*-
"""
Nhận diện mệnh giá tiền Việt Nam bằng xử lý ảnh cổ điển (KHÔNG dùng AI / học máy).

Các bước:
  1. Tiền xử lý   : thu nhỏ ảnh, cân bằng trắng theo vùng sáng trung tính.
  2. Định vị      : tìm vùng hình chữ nhật có tỉ lệ dài/rộng giống tờ tiền (~2.0-2.6),
                    nắn phối cảnh về kích thước chuẩn 640x288.
  3. Màu sắc      : histogram kênh Hue (HSV) có trọng số độ bão hòa, so với màu chủ đạo
                    của từng mệnh giá (có sẵn) hoặc màu học từ ảnh mẫu.
  4. Hình học     : điểm đặc trưng ORB + so khớp với ảnh mẫu, lọc nhiễu bằng RANSAC.
  5. Kết hợp      : cộng điểm màu và điểm đặc trưng, chọn mệnh giá điểm cao nhất.

Dùng dòng lệnh:
  python recognizer.py anh1.jpg anh2.jpg          # nhận diện, lưu ảnh *_ketqua.jpg
  python recognizer.py --eval thu_muc_kiem_tra    # đánh giá độ chính xác
  python recognizer.py --rebuild                  # tính lại toàn bộ cache ảnh mẫu
"""
from __future__ import annotations

import os
import sys
import glob
import pickle
from dataclasses import dataclass, field

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# Cấu hình
# --------------------------------------------------------------------------- #
DENOMINATIONS = [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000]

# Màu chủ đạo của tiền polymer (Hue theo thang OpenCV 0..179): (tâm, độ rộng)
# Đây là giá trị mặc định khi chưa có ảnh mẫu. Có ảnh mẫu thì màu sẽ được học lại.
COLOR_PROTOTYPES = {
    10000:  [(22, 6)],             # vàng nâu / vàng đất
    20000:  [(110, 7)],            # xanh dương đậm
    50000:  [(160, 9)],            # hồng tím
    100000: [(48, 12)],            # xanh lá ô liu
    200000: [(9, 5), (176, 4)],    # đỏ cam / nâu đỏ
    500000: [(94, 6)],             # xanh lơ / xanh ngọc
}

NOTE_W, NOTE_H = 640, 288          # kích thước chuẩn sau khi nắn (tỉ lệ ~2.22)
MAX_SIDE = 1000                    # ảnh đầu vào được thu nhỏ về cạnh dài tối đa
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

W_FEATURE, W_COLOR = 0.65, 0.35    # trọng số kết hợp khi có ảnh mẫu
INLIERS_FULL = 40                  # số inlier ORB coi như khớp 100%
MIN_CONFIDENCE = 0.30              # dưới ngưỡng này -> "không chắc chắn"


def format_vnd(v):
    return f"{v:,}".replace(",", ".") + " đ"


# --------------------------------------------------------------------------- #
# Đọc / ghi ảnh hỗ trợ đường dẫn có dấu tiếng Việt trên Windows
# --------------------------------------------------------------------------- #
def imread_unicode(path):
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path, img):
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(path)
    return ok


# --------------------------------------------------------------------------- #
# Tiền xử lý
# --------------------------------------------------------------------------- #
def resize_max(img, max_side=MAX_SIDE):
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def white_balance(img):
    """Cân bằng trắng "white-patch": lấy các điểm sáng nhất, gần trung tính (lề trắng,
    cửa sổ trong suốt, giấy nền...) làm chuẩn trắng. Không làm mất màu chủ đạo của tờ tiền
    như phương pháp gray-world. Hệ số hiệu chỉnh bị giới hạn để tránh méo màu."""
    f = img.reshape(-1, 3).astype(np.float32)
    lum = f.mean(axis=1)
    chroma = f.max(axis=1) - f.min(axis=1)
    cand = (lum > np.percentile(lum, 90)) & (chroma < 0.35 * lum + 1) & (lum < 250)
    if cand.sum() < 50:
        return img
    ref = f[cand].mean(axis=0) + 1e-6
    gain = np.clip(ref.mean() / ref, 0.75, 1.35)
    return np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)



# --------------------------------------------------------------------------- #
# Định vị tờ tiền
# --------------------------------------------------------------------------- #
def _order_points(pts):
    pts = np.asarray(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / float(aw * ah + bw * bh - inter + 1e-6)


def _foreground_masks(img):
    """Tạo vài mặt nạ ứng viên: (1) khác màu nền viền ảnh, (2) cạnh Canny."""
    h, w = img.shape[:2]
    masks = []

    # (1) Khác biệt so với màu nền (ước lượng từ dải viền ảnh) trong không gian Lab
    lab = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    b = max(4, min(h, w) // 40)
    border = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
                             lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    dist = np.linalg.norm(lab - bg, axis=2)
    dist = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, m1 = cv2.threshold(dist, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    m1 = cv2.morphologyEx(m1, cv2.MORPH_CLOSE, k, iterations=2)
    m1 = cv2.morphologyEx(m1, cv2.MORPH_OPEN, k, iterations=1)
    masks.append(m1)

    # (2) Cạnh Canny + đóng hình thái học, lấp kín các contour
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 30, 100)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=2)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    m2 = np.zeros_like(gray)
    cv2.drawContours(m2, cnts, -1, 255, thickness=cv2.FILLED)
    m2 = cv2.erode(m2, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=2)
    masks.append(m2)
    return masks


def find_notes(img, max_notes=10):
    """Trả về danh sách hình chữ nhật xoay (cv2.minAreaRect) bao các tờ tiền."""
    h, w = img.shape[:2]
    area = float(h * w)
    cands = []
    for mask in _foreground_masks(img):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            a = cv2.contourArea(c)
            if a < 0.02 * area or a > 0.97 * area:
                continue
            rect = cv2.minAreaRect(c)
            rw, rh = rect[1]
            if min(rw, rh) < 20:
                continue
            ratio = max(rw, rh) / min(rw, rh)
            fill = a / (rw * rh)
            if 1.7 <= ratio <= 2.9 and fill >= 0.80:
                score = fill - abs(ratio - 2.25) * 0.2
                cands.append((score, a, rect))

    # Loại bỏ trùng lặp (non-maximum suppression)
    cands.sort(key=lambda x: (-x[0], -x[1]))
    keep = []
    for score, a, rect in cands:
        bb = cv2.boundingRect(cv2.boxPoints(rect).astype(np.int32))
        if all(_iou(bb, cv2.boundingRect(cv2.boxPoints(r).astype(np.int32))) < 0.3 for r in keep):
            keep.append(rect)
        if len(keep) >= max_notes:
            break
    return keep


def warp_note(img, rect_or_pts):
    """Nắn tờ tiền về ảnh NOTE_W x NOTE_H (cạnh dài nằm ngang)."""
    if isinstance(rect_or_pts, tuple):
        pts = cv2.boxPoints(rect_or_pts)
    else:
        pts = np.asarray(rect_or_pts, dtype=np.float32)
    tl, tr, br, bl = _order_points(pts)
    if np.linalg.norm(tr - tl) < np.linalg.norm(bl - tl):     # đang dựng đứng -> xoay
        tl, tr, br, bl = tr, br, bl, tl
    src = np.array([tl, tr, br, bl], dtype=np.float32)
    dst = np.array([[0, 0], [NOTE_W - 1, 0], [NOTE_W - 1, NOTE_H - 1], [0, NOTE_H - 1]],
                   dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (NOTE_W, NOTE_H)), src


def full_image_quad(img):
    h, w = img.shape[:2]
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)


# --------------------------------------------------------------------------- #
# Đặc trưng màu
# --------------------------------------------------------------------------- #
def hue_histogram(note_bgr):
    """Histogram Hue 180 bin, trọng số theo độ bão hòa, bỏ điểm quá tối/quá trắng."""
    hsv = cv2.cvtColor(note_bgr, cv2.COLOR_BGR2HSV)
    hch, s, v = hsv[..., 0].ravel(), hsv[..., 1].ravel().astype(np.float32), hsv[..., 2].ravel()
    m = (s > 45) & (v > 40) & (v < 252)
    hist = np.bincount(hch[m], weights=s[m], minlength=180)[:180].astype(np.float32)
    # làm mượt vòng tròn (Hue là đại lượng tuần hoàn)
    k = cv2.getGaussianKernel(9, 2).ravel()
    padded = np.concatenate([hist[-4:], hist, hist[:4]])
    hist = np.convolve(padded, k, mode="valid")
    tot = hist.sum()
    return hist / tot if tot > 0 else hist


def prototype_histogram(peaks):
    x = np.arange(180, dtype=np.float32)
    hist = np.zeros(180, np.float32)
    for mu, sigma in peaks:
        d = np.minimum(np.abs(x - mu), 180 - np.abs(x - mu))
        hist += np.exp(-0.5 * (d / sigma) ** 2)
    return hist / hist.sum()


def histogram_intersection(a, b):
    return float(np.minimum(a, b).sum())


# --------------------------------------------------------------------------- #
# Cơ sở dữ liệu ảnh mẫu
# --------------------------------------------------------------------------- #
@dataclass
class Template:
    denom: int
    path: str
    kps: list
    des: np.ndarray


def _prep_gray(note_bgr):
    gray = cv2.cvtColor(note_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


class TemplateDB:
    """Thư mục mẫu: templates/<mệnh giá>/*.jpg  (ví dụ templates/500000/mat_truoc.jpg)."""

    def __init__(self, folder):
        self.folder = folder
        self.orb = cv2.ORB_create(nfeatures=1500, scaleFactor=1.2, nlevels=8)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.templates = []
        self.color_refs = {}
        self.reload()

    # ---------- Bộ nhớ đệm (cache) ----------
    # Lưu keypoint/descriptor ORB và histogram màu của từng ảnh mẫu vào file
    # templates/_cache.pkl. Lần chạy sau chỉ tính lại những ảnh mới hoặc đã sửa.
    CACHE_NAME = "_cache.pkl"
    CACHE_VERSION = 1

    def _cache_key(self):
        # Đổi tham số xử lý thì cache cũ tự động bị bỏ
        return (self.CACHE_VERSION, NOTE_W, NOTE_H, MAX_SIDE,
                self.orb.getMaxFeatures(), self.orb.getScaleFactor(), self.orb.getNLevels())

    @staticmethod
    def _kps_to_array(kps):
        return np.array([(k.pt[0], k.pt[1], k.size, k.angle, k.response, k.octave, k.class_id)
                         for k in kps], dtype=np.float32).reshape(-1, 7)

    @staticmethod
    def _array_to_kps(arr):
        return [cv2.KeyPoint(float(x), float(y), float(sz), float(a), float(r), int(o), int(c))
                for x, y, sz, a, r, o, c in arr]

    def _load_cache(self, path):
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            if data.get("key") == self._cache_key():
                return data.get("files", {})
        except Exception:
            pass
        return {}

    def _save_cache(self, path, files):
        try:
            tmp = path + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump({"key": self._cache_key(), "files": files}, f,
                            protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, path)
        except OSError as e:
            print(f"Không ghi được cache: {e}")

    def _compute_entry(self, path, denom):
        img = imread_unicode(path)
        if img is None:
            return None
        note = extract_main_note(img)
        kps, des = self.orb.detectAndCompute(_prep_gray(note), None)
        return {"denom": denom, "kps": self._kps_to_array(kps or []),
                "des": des, "hist": hue_histogram(note)}

    def reload(self, force=False):
        """Nạp ảnh mẫu. Dùng cache nếu ảnh không đổi; force=True để tính lại toàn bộ."""
        os.makedirs(self.folder, exist_ok=True)
        cache_path = os.path.join(self.folder, self.CACHE_NAME)
        old = {} if force else self._load_cache(cache_path)
        files, changed = {}, False
        self.last_reload_stats = {"cache": 0, "computed": 0}

        for d in sorted(os.listdir(self.folder)):
            sub = os.path.join(self.folder, d)
            if not os.path.isdir(sub) or not d.isdigit() or int(d) not in DENOMINATIONS:
                continue
            denom = int(d)
            for p in sorted(glob.glob(os.path.join(sub, "*"))):
                if not p.lower().endswith(IMG_EXTS):
                    continue
                rel = os.path.relpath(p, self.folder).replace(os.sep, "/")
                st = os.stat(p)
                sig = (st.st_mtime_ns, st.st_size)
                entry = old.get(rel)
                if entry is not None and entry.get("sig") == sig and entry.get("denom") == denom:
                    self.last_reload_stats["cache"] += 1
                else:
                    entry = self._compute_entry(p, denom)
                    if entry is None:
                        continue
                    entry["sig"] = sig
                    changed = True
                    self.last_reload_stats["computed"] += 1
                files[rel] = entry

        if changed or set(files) != set(old):
            self._save_cache(cache_path, files)

        self.templates = []
        sums = {}
        for rel, e in files.items():
            if e["des"] is not None and len(e["kps"]) >= 10:
                self.templates.append(Template(e["denom"], os.path.join(self.folder, rel),
                                               self._array_to_kps(e["kps"]), e["des"]))
            sums.setdefault(e["denom"], []).append(e["hist"])
        self.color_refs = {}
        for denom, hs in sums.items():
            h = np.mean(hs, axis=0)
            self.color_refs[denom] = h / h.sum() if h.sum() > 0 else h

    def counts(self):
        c = {}
        for t in self.templates:
            c[t.denom] = c.get(t.denom, 0) + 1
        return c

    def color_reference(self, denom):
        if denom in self.color_refs:
            return self.color_refs[denom]
        if denom in COLOR_PROTOTYPES:
            return prototype_histogram(COLOR_PROTOTYPES[denom])
        return None

    def match(self, note_bgr):
        """Trả về {mệnh giá: số inlier lớn nhất} khi so khớp ORB với các ảnh mẫu."""
        result = {}
        if not self.templates:
            return result
        kps, des = self.orb.detectAndCompute(_prep_gray(note_bgr), None)
        if des is None or len(kps) < 10:
            return result
        for t in self.templates:
            pairs = self.matcher.knnMatch(des, t.des, k=2)
            good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
            inl = 0
            if len(good) >= 8:
                src = np.float32([kps[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst = np.float32([t.kps[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 6.0)
                if H is not None and mask is not None:
                    det = np.linalg.det(H[:2, :2])
                    if 0.2 < abs(det) < 5:          # loại phép biến đổi suy biến
                        inl = int(mask.sum())
            result[t.denom] = max(result.get(t.denom, 0), inl)
        return result


def extract_main_note(img):
    img = resize_max(img)
    rects = find_notes(img, max_notes=1)
    note, _ = warp_note(white_balance(img), rects[0] if rects else full_image_quad(img))
    return note


# --------------------------------------------------------------------------- #
# Nhận diện
# --------------------------------------------------------------------------- #
@dataclass
class NoteResult:
    denom: int | None
    confidence: float
    quad: np.ndarray                       # 4 góc trong ảnh đã thu nhỏ
    scores: dict = field(default_factory=dict)
    color_scores: dict = field(default_factory=dict)
    inliers: dict = field(default_factory=dict)
    note_img: np.ndarray | None = None

    @property
    def label(self):
        return format_vnd(self.denom) if self.denom else "Không chắc chắn"


class BanknoteRecognizer:
    def __init__(self, template_folder=None):
        if template_folder is None:
            template_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
        self.db = TemplateDB(template_folder)

    def classify_note(self, note_bgr, use_features=True):
        note = note_bgr
        hist = hue_histogram(note)

        cands = sorted(set(COLOR_PROTOTYPES) | set(self.db.color_refs))
        raw = {d: histogram_intersection(hist, self.db.color_reference(d)) for d in cands}
        tot = sum(raw.values()) or 1.0
        color = {d: v / tot for d, v in raw.items()}

        inliers = self.db.match(note) if use_features else {}
        if inliers:
            scores = {}
            for d in set(color) | set(inliers):
                f = min(inliers.get(d, 0) / INLIERS_FULL, 1.0)
                scores[d] = W_FEATURE * f + W_COLOR * color.get(d, 0.0)
        else:
            scores = dict(color)

        best = max(scores, key=scores.get) if scores else None
        conf = scores.get(best, 0.0)
        # Có mẫu và khớp đặc trưng rất mạnh thì tin tưởng cao
        if inliers and inliers.get(best, 0) >= 25:
            conf = max(conf, 0.9)
        # Chỉ dùng màu: độ tin cậy dựa trên khoảng cách giữa hạng 1 và hạng 2
        if not inliers and len(scores) > 1:
            top = sorted(scores.values(), reverse=True)
            conf = 1.0 - top[1] / (top[0] + 1e-6)
        if conf < MIN_CONFIDENCE or sum(raw.values()) < 0.05:
            best = None
        return best, conf, scores, color, inliers

    def recognize(self, img, use_features=True, max_notes=10):
        """Nhận diện tất cả tờ tiền trong ảnh. Trả về (ảnh đã thu nhỏ, danh sách NoteResult)."""
        img = resize_max(img)
        rects = find_notes(img, max_notes=max_notes)
        regions = rects if rects else [full_image_quad(img)]
        balanced = white_balance(img)
        results = []
        for r in regions:
            note, quad = warp_note(balanced, r)
            d, conf, scores, color, inl = self.classify_note(note, use_features)
            results.append(NoteResult(d, conf, quad, scores, color, inl, note))
        return img, results


# --------------------------------------------------------------------------- #
# Vẽ kết quả
# --------------------------------------------------------------------------- #
def ascii_label(r):
    return (f"{r.denom:,}".replace(",", ".") + " VND") if r.denom else "?"


def draw_results(img, results):
    out = img.copy()
    for r in results:
        color = (0, 200, 0) if r.denom else (0, 0, 255)
        q = r.quad.astype(np.int32)
        cv2.polylines(out, [q], True, color, 3)
        x, y = q[:, 0].min(), max(q[:, 1].min() - 10, 25)
        text = f"{ascii_label(r)} ({r.confidence * 100:.0f}%)"
        cv2.putText(out, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5)
        cv2.putText(out, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    total = sum(r.denom for r in results if r.denom)
    t = f"Tong: {total:,}".replace(",", ".") + " VND"
    cv2.putText(out, t, (10, out.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5)
    cv2.putText(out, t, (10, out.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return out


# --------------------------------------------------------------------------- #
# Dòng lệnh
# --------------------------------------------------------------------------- #
def _evaluate(folder, rec):
    ok = n = 0
    for d in sorted(os.listdir(folder)):
        sub = os.path.join(folder, d)
        if not (os.path.isdir(sub) and d.isdigit()):
            continue
        for p in sorted(glob.glob(os.path.join(sub, "*"))):
            if not p.lower().endswith(IMG_EXTS):
                continue
            img = imread_unicode(p)
            if img is None:
                continue
            _, res = rec.recognize(img, max_notes=1)
            pred = res[0].denom if res else None
            n += 1
            ok += int(pred == int(d))
            mark = "OK " if pred == int(d) else "SAI"
            print(f"[{mark}] {os.path.basename(p):30s} thật={int(d):>7}  đoán={pred}")
    if n:
        print(f"\nĐộ chính xác: {ok}/{n} = {ok / n * 100:.1f}%")


def main(argv):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if not argv:
        print(__doc__)
        return
    rec = BanknoteRecognizer()
    st = rec.db.last_reload_stats
    print(f"Đã nạp {len(rec.db.templates)} ảnh mẫu: {rec.db.counts()} "
          f"(từ cache: {st['cache']}, tính mới: {st['computed']})")
    if argv[0] == "--rebuild":
        rec.db.reload(force=True)
        print("Đã tạo lại cache.")
        return
    if argv[0] == "--eval":
        _evaluate(argv[1], rec)
        return
    for p in argv:
        img = imread_unicode(p)
        if img is None:
            print(f"Không đọc được ảnh: {p}")
            continue
        small, res = rec.recognize(img)
        print(f"\n{p}: {len(res)} tờ")
        for i, r in enumerate(res, 1):
            top = sorted(r.scores.items(), key=lambda kv: -kv[1])[:3]
            tops = ", ".join(f"{k}:{v:.2f}" for k, v in top)
            print(f"  #{i}: {r.label:>16}  tin cậy {r.confidence * 100:5.1f}%   [{tops}]")
        total = sum(r.denom for r in res if r.denom)
        print(f"  Tổng: {format_vnd(total)}")
        outp = os.path.splitext(p)[0] + "_ketqua.jpg"
        imwrite_unicode(outp, draw_results(small, res))
        print(f"  Đã lưu: {outp}")


if __name__ == "__main__":
    main(sys.argv[1:])
