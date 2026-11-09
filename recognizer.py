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


def _rect_corners(w, h):
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)


def warp_note(img, rect_or_pts, size=(NOTE_W, NOTE_H)):
    """Nắn tờ tiền về ảnh size = (rộng, cao), mặc định NOTE_W x NOTE_H (cạnh dài nằm ngang)."""
    if isinstance(rect_or_pts, tuple):
        pts = cv2.boxPoints(rect_or_pts)
    else:
        pts = np.asarray(rect_or_pts, dtype=np.float32)
    tl, tr, br, bl = _order_points(pts)
    if np.linalg.norm(tr - tl) < np.linalg.norm(bl - tl):     # đang dựng đứng -> xoay
        tl, tr, br, bl = tr, br, bl, tl
    src = np.array([tl, tr, br, bl], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, _rect_corners(*size))
    return cv2.warpPerspective(img, M, size), src


def full_image_quad(img):
    h, w = img.shape[:2]
    return _rect_corners(w, h)


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


def _scan_template_files(folder):
    """Duyệt templates/<mệnh giá>/*.<ảnh>. Trả về (đường dẫn tương đối, đường dẫn, mệnh giá, chữ ký)."""
    for d in sorted(os.listdir(folder)):
        sub = os.path.join(folder, d)
        if not os.path.isdir(sub) or not d.isdigit() or int(d) not in DENOMINATIONS:
            continue
        for p in sorted(glob.glob(os.path.join(sub, "*"))):
            if p.lower().endswith(IMG_EXTS):
                st = os.stat(p)
                rel = os.path.relpath(p, folder).replace(os.sep, "/")
                yield rel, p, int(d), (st.st_mtime_ns, st.st_size)


def _load_template_entries(folder, cache_name, key, compute, force=False):
    """Tính đặc trưng cho mọi ảnh mẫu, có bộ nhớ đệm.

    Kết quả của compute(path, denom) được lưu vào folder/cache_name cùng chữ ký file
    (mtime, size); lần sau chỉ tính lại ảnh mới hoặc đã sửa. Đổi `key` (tham số xử lý)
    thì toàn bộ cache cũ bị bỏ. Trả về ({đường dẫn tương đối: entry}, thống kê).
    """
    os.makedirs(folder, exist_ok=True)
    cache_path = os.path.join(folder, cache_name)
    old = {}
    if not force:
        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            if data.get("key") == key:
                old = data.get("files", {})
        except Exception:
            pass

    files, stats = {}, {"cache": 0, "computed": 0}
    for rel, path, denom, sig in _scan_template_files(folder):
        entry = old.get(rel)
        if entry is not None and entry.get("sig") == sig and entry.get("denom") == denom:
            stats["cache"] += 1
        else:
            entry = compute(path, denom)
            if entry is None:
                continue
            entry["sig"], entry["denom"] = sig, denom
            stats["computed"] += 1
        files[rel] = entry

    if stats["computed"] or set(files) != set(old):
        try:
            tmp = cache_path + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump({"key": key, "files": files}, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, cache_path)
        except OSError as e:
            print(f"Không ghi được cache: {e}")
    return files, stats


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

    def _compute_entry(self, path, denom):
        img = imread_unicode(path)
        if img is None:
            return None
        note = extract_main_note(img)
        kps, des = self.orb.detectAndCompute(_prep_gray(note), None)
        return {"kps": self._kps_to_array(kps or []), "des": des, "hist": hue_histogram(note)}

    def reload(self, force=False):
        """Nạp ảnh mẫu. Dùng cache nếu ảnh không đổi; force=True để tính lại toàn bộ."""
        files, self.last_reload_stats = _load_template_entries(
            self.folder, self.CACHE_NAME, self._cache_key(), self._compute_entry, force)

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


def extract_main_note(img, size=(NOTE_W, NOTE_H)):
    img = resize_max(img)
    rects = find_notes(img, max_notes=1)
    note, _ = warp_note(white_balance(img), rects[0] if rects else full_image_quad(img), size)
    return note


# --------------------------------------------------------------------------- #
# Nhận diện NHIỀU tờ, kể cả tờ bị đè / bị che một phần
# --------------------------------------------------------------------------- #
# Cách tách nền rồi tìm hình chữ nhật (find_notes) thất bại khi các tờ chồng lên nhau:
# chúng dính thành một khối không còn tỉ lệ của tờ tiền. Ở đây mỗi tờ được tìm trực tiếp
# bằng điểm đặc trưng cục bộ, nên chỉ cần PHẦN CÒN NHÌN THẤY có đủ điểm khớp:
#
#   1. Trích đặc trưng (SIFT hoặc ORB) trên toàn ảnh và trên mọi ảnh mẫu.
#   2. Ratio test THEO LỚP (Lowe 2004): giữ một cặp khớp chỉ khi láng giềng gần nhất
#      gần hơn hẳn láng giềng gần nhất thuộc MỆNH GIÁ KHÁC. Chi tiết chung của mọi tờ
#      (chân dung, quốc huy, dòng quốc hiệu) không phân biệt được nên tự bị loại.
#   3. RANSAC lặp (sequential RANSAC): trong mọi ảnh mẫu, lấy homography có nhiều inlier
#      nhất -> nhận một tờ -> xóa các điểm đã dùng -> lặp lại cho tờ tiếp theo.
#      Chiếu 4 góc ảnh mẫu qua homography được khung ĐẦY ĐỦ của tờ, kể cả phần bị che.
#   4. Kiểm tra hình học (không lật gương, tứ giác lồi, tỉ lệ cạnh hợp lý, các điểm khớp
#      trải đủ rộng) và loại phát hiện trùng lặp.
FEATURE_KIND = "sift"              # "sift": chính xác hơn | "orb": nhanh hơn
DETECT_MAX_SIDE = 1400             # cạnh dài của ảnh khi trích đặc trưng để dò nhiều tờ
TPL_W, TPL_H = 1000, 450           # kích thước ảnh mẫu (đã nắn) khi trích đặc trưng
TPL_FEATURES = 8000                # số điểm đặc trưng tối đa trên mỗi ảnh mẫu
KNN_K = 8                          # số láng giềng xét trong ratio test theo lớp
CLASS_RATIO = 0.85                 # ngưỡng ratio test theo lớp
MIN_INLIERS = 15                   # số inlier tối thiểu để chấp nhận một tờ
RANSAC_REPROJ = 5.0                # sai số chiếu lại tối đa (pixel, ảnh truy vấn)
MIN_SPREAD = 0.04                  # inlier phủ >= 4% diện tích ảnh mẫu (bao lồi) ...
MIN_SPREAD_THIN = 0.01             # ... hoặc >= 1% (tờ chỉ lộ một dải mỏng) nhưng khi đó
THIN_INLIERS = 30                  #     cần >= 30 inlier: vùng càng nhỏ càng cần nhiều bằng chứng
_RANSAC_METHOD = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
_TPL_CORNERS = _rect_corners(TPL_W, TPL_H).reshape(-1, 1, 2)


def _make_features(kind, n):
    if kind == "sift":
        return cv2.SIFT_create(nfeatures=n, contrastThreshold=0.03)
    if kind == "orb":
        return cv2.ORB_create(nfeatures=n or 15000, scaleFactor=1.2, nlevels=8, fastThreshold=10)
    raise ValueError(f"Loại đặc trưng không hỗ trợ: {kind}")


def _quad_iou(a, b):
    a = np.asarray(a, np.float32).reshape(-1, 2)
    b = np.asarray(b, np.float32).reshape(-1, 2)
    inter, _ = cv2.intersectConvexConvex(a, b)
    union = cv2.contourArea(a) + cv2.contourArea(b) - inter
    return inter / union if union > 0 else 0.0


def _plausible(H, quad, img_area, inl_tpl_pts):
    """Kiểm tra homography có tương ứng với một tờ tiền thật hay không."""
    if np.linalg.det(H[:2, :2]) <= 0:                   # lật gương: tờ tiền thật không thể
        return False
    q = quad.reshape(-1, 1, 2).astype(np.float32)
    if not cv2.isContourConvex(q):
        return False
    if not 0.002 * img_area <= cv2.contourArea(q) <= 2.0 * img_area:
        return False
    e = np.linalg.norm(quad - np.roll(quad, -1, axis=0), axis=1)   # trên, phải, dưới, trái
    if e.min() < 15:
        return False
    if not 1.5 <= (e[0] + e[2]) / (e[1] + e[3]) <= 3.2:              # tỉ lệ dài/rộng
        return False
    if max(e[0], e[2]) > 2 * min(e[0], e[2]) or max(e[1], e[3]) > 2 * min(e[1], e[3]):
        return False                                                 # phối cảnh quá méo
    hull = cv2.convexHull(inl_tpl_pts.reshape(-1, 1, 2).astype(np.float32))
    spread = cv2.contourArea(hull) / (TPL_W * TPL_H)
    return spread >= MIN_SPREAD or (spread >= MIN_SPREAD_THIN and len(inl_tpl_pts) >= THIN_INLIERS)


DUP_CORNER_DIST = 0.15             # 4 góc lệch trung bình < 15% cạnh dài -> cùng một tờ


def _is_duplicate(det, accepted):
    """Phát hiện mới là trùng nếu 4 góc của nó gần như trùng 4 góc một tờ đã nhận (cùng một
    tờ khớp với ảnh mẫu khác). Không dùng IoU: các tờ xòe quạt nằm cạnh nhau có IoU rất cao
    nhưng là các tờ khác nhau. So mọi phép xoay vòng thứ tự góc (tờ lật ngược 180 độ)."""
    q = det["quad"]
    side = np.linalg.norm(q[1] - q[0])
    for a in accepted:
        d = min(np.linalg.norm(np.roll(a["quad"], k, axis=0) - q, axis=1).mean() for k in range(4))
        if d < DUP_CORNER_DIST * side:
            return True
    return False


class FeatureIndex:
    """Chỉ mục đặc trưng của mọi ảnh mẫu, dùng để dò nhiều tờ (kể cả bị đè) trong một ảnh."""

    def __init__(self, folder, kind=FEATURE_KIND):
        self.folder, self.kind = folder, kind
        self.tpl_extractor = _make_features(kind, TPL_FEATURES)
        self.query_extractor = _make_features(kind, 0)
        self.reload()

    def _cache_key(self):
        return ("features", 1, self.kind, TPL_W, TPL_H, TPL_FEATURES, MAX_SIDE)

    def _compute_entry(self, path, denom):
        img = imread_unicode(path)
        if img is None:
            return None
        note = extract_main_note(img, (TPL_W, TPL_H))
        kps, des = self.tpl_extractor.detectAndCompute(_prep_gray(note), None)
        if des is None:
            return None
        return {"pts": np.float32([k.pt for k in kps]).reshape(-1, 2), "des": des}

    def reload(self, force=False):
        files, self.last_reload_stats = _load_template_entries(
            self.folder, f"_cache_{self.kind}.pkl", self._cache_key(), self._compute_entry, force)
        entries = [(rel, e) for rel, e in files.items() if len(e["pts"]) >= MIN_INLIERS]
        self.tpls = [{"denom": e["denom"], "path": rel, "pts": e["pts"]} for rel, e in entries]
        self.matcher = None
        if not entries:
            return
        descs = [e["des"] for _, e in entries]
        self.owner_tpl = np.concatenate([np.full(len(d), i) for i, d in enumerate(descs)])
        self.owner_denom = np.array([self.tpls[i]["denom"] for i in self.owner_tpl])
        self.local_idx = np.concatenate([np.arange(len(d)) for d in descs])
        all_des = np.vstack(descs)
        if self.kind == "sift":
            self.matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
            all_des = all_des.astype(np.float32)
        else:
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.matcher.add([all_des])
        self.matcher.train()

    # ------------------------------------------------------------------ #
    def detect(self, img, max_notes=10):
        """Dò mọi tờ trong ảnh BGR. Trả về danh sách dict(denom, quad 4x2 theo toạ độ img,
        inliers, tpl) — quad theo thứ tự góc trên-trái, trên-phải, dưới-phải, dưới-trái của tờ."""
        if self.matcher is None:
            return []
        kps, des = self.query_extractor.detectAndCompute(_prep_gray(img), None)
        if des is None or len(kps) < MIN_INLIERS:
            return []
        if self.kind == "sift":
            des = des.astype(np.float32)
        qpts = np.float32([k.pt for k in kps]).reshape(-1, 2)
        matches = self._class_ratio_matches(des)
        return self._sequential_ransac(qpts, matches, img.shape[0] * img.shape[1], max_notes)

    def _class_ratio_matches(self, des):
        """Ratio test theo lớp. Trả về {chỉ số ảnh mẫu: (chỉ số điểm ảnh truy vấn, chỉ số điểm mẫu)}."""
        per = {}
        for nbrs in self.matcher.knnMatch(des, k=KNN_K):
            if not nbrs:
                continue
            c0 = self.owner_denom[nbrs[0].trainIdx]
            d_other = next((n.distance for n in nbrs if self.owner_denom[n.trainIdx] != c0), None)
            limit = CLASS_RATIO * d_other if d_other is not None else np.inf
            seen = set()
            for n in nbrs:              # mọi ảnh mẫu cùng mệnh giá đủ gần đều được nhận
                if n.distance >= limit:
                    break
                t = int(self.owner_tpl[n.trainIdx])
                if t not in seen:
                    seen.add(t)
                    q, l = per.setdefault(t, ([], []))
                    q.append(n.queryIdx)
                    l.append(int(self.local_idx[n.trainIdx]))
        return {t: (np.array(q), np.array(l)) for t, (q, l) in per.items()}

    def _fit(self, t, q_idx, t_idx, qpts, img_area):
        src = self.tpls[t]["pts"][t_idx].reshape(-1, 1, 2)
        dst = qpts[q_idx].reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, _RANSAC_METHOD, RANSAC_REPROJ,
                                     maxIters=4000, confidence=0.999)
        if H is None or mask is None:
            return None
        mask = mask.ravel().astype(bool)
        if mask.sum() < MIN_INLIERS:
            return None
        quad = cv2.perspectiveTransform(_TPL_CORNERS, H).reshape(4, 2)
        if not _plausible(H, quad, img_area, self.tpls[t]["pts"][t_idx[mask]]):
            return None
        return {"tpl": t, "denom": self.tpls[t]["denom"], "quad": quad,
                "inliers": int(mask.sum()), "q_inl": q_idx[mask]}

    def _sequential_ransac(self, qpts, matches, img_area, max_notes):
        alive = np.ones(len(qpts), bool)        # điểm ảnh truy vấn chưa thuộc tờ nào
        cache = {}                              # ảnh mẫu -> (số điểm còn sống lúc fit, kết quả)
        accepted = []
        while len(accepted) < max_notes:
            best = None
            for t, (q_idx, t_idx) in matches.items():
                live = alive[q_idx]
                n_live = int(live.sum())
                if n_live < MIN_INLIERS:
                    continue
                if t not in cache or cache[t][0] != n_live:    # chỉ fit lại khi mất điểm
                    cache[t] = (n_live, self._fit(t, q_idx[live], t_idx[live], qpts, img_area))
                r = cache[t][1]
                if r is not None and (best is None or r["inliers"] > best["inliers"]):
                    best = r
            if best is None:
                break
            alive[best["q_inl"]] = False
            if not _is_duplicate(best, accepted):
                accepted.append(best)
        return accepted


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
    def __init__(self, template_folder=None, feature_kind=FEATURE_KIND):
        if template_folder is None:
            template_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
        self.db = TemplateDB(template_folder)
        self.index = FeatureIndex(template_folder, feature_kind)

    def reload(self, force=False):
        self.db.reload(force)
        self.index.reload(force)

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

    def recognize(self, img, use_features=True, max_notes=10, method="multi"):
        """Nhận diện tất cả tờ tiền trong ảnh. Trả về (ảnh đã thu nhỏ, danh sách NoteResult).

        method="multi"  : dò từng tờ bằng đặc trưng + RANSAC lặp (nhận được tờ bị đè),
                          phần còn lại bổ sung bằng tách nền + màu.
        method="classic": cách cũ — tách nền, tìm hình chữ nhật, phân loại màu + ORB.
        use_features=False: chỉ dùng tách nền + màu (không cần ảnh mẫu).
        """
        small = resize_max(img)
        if not use_features or method == "classic" or self.index.matcher is None:
            return small, self._recognize_classic(small, use_features, max_notes)

        big = resize_max(img, DETECT_MAX_SIDE)
        scale = small.shape[1] / big.shape[1]
        balanced = white_balance(small)
        results = []
        for d in self.index.detect(big, max_notes):
            quad = (d["quad"] * scale).astype(np.float32)
            M = cv2.getPerspectiveTransform(quad, _rect_corners(NOTE_W, NOTE_H))
            note = cv2.warpPerspective(balanced, M, (NOTE_W, NOTE_H))
            conf = min(1.0, 0.5 + d["inliers"] / 100)
            results.append(NoteResult(d["denom"], conf, quad, {d["denom"]: conf}, {},
                                      {d["denom"]: d["inliers"]}, note))
        results += self._leftover_regions(small, balanced, results, max_notes - len(results))
        return small, results

    def _recognize_classic(self, small, use_features, max_notes):
        rects = find_notes(small, max_notes=max_notes)
        regions = rects if rects else [full_image_quad(small)]
        balanced = white_balance(small)
        results = []
        for r in regions:
            note, quad = warp_note(balanced, r)
            d, conf, scores, color, inl = self.classify_note(note, use_features)
            results.append(NoteResult(d, conf, quad, scores, color, inl, note))
        return results

    def _leftover_regions(self, small, balanced, found, n):
        """Vùng hình chữ nhật tìm bằng tách nền mà không trùng tờ nào đã dò được
        (ví dụ tờ chưa có ảnh mẫu): phân loại bằng màu."""
        if n <= 0:
            return []
        covered = np.zeros(small.shape[:2], np.uint8)
        for r in found:
            cv2.fillConvexPoly(covered, r.quad.astype(np.int32), 255)
        out = []
        for rect in find_notes(small, max_notes=n):
            box = np.zeros_like(covered)
            cv2.fillConvexPoly(box, cv2.boxPoints(rect).astype(np.int32), 255)
            area = np.count_nonzero(box)
            if area == 0 or np.count_nonzero(box & covered) > 0.2 * area:
                continue
            note, quad = warp_note(balanced, rect)
            d, conf, scores, color, inl = self.classify_note(note, use_features=False)
            out.append(NoteResult(d, conf, quad, scores, color, inl, note))
        return out


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
        rec.reload(force=True)
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
