# -*- coding: utf-8 -*-
"""
Giao diện nhận diện mệnh giá tiền Việt Nam (xử lý ảnh cổ điển, không dùng AI).

Chạy:  python main.py
"""
import os
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
from PIL import Image, ImageTk

import recognizer as R

VIEW_W, VIEW_H = 820, 560
WEBCAM_EVERY = 3            # nhận diện 1 lần mỗi N khung hình để giữ mượt


class App:
    def __init__(self, root):
        self.root = root
        root.title("Nhận diện mệnh giá tiền Việt Nam")
        root.geometry("1200x640")
        root.minsize(900, 560)
        self.rec = R.BanknoteRecognizer()
        self.cap = None
        self.frame_no = 0
        self.last_img = None        # ảnh gốc (BGR) đang hiển thị / khung hình hiện tại
        self.last_results = []
        self._photo = None

        # ----- Bảng điều khiển (pack trước để luôn hiển thị bên phải) -----
        right = ttk.Frame(root, padding=8)
        right.pack(side=tk.RIGHT, fill=tk.Y)

        # ----- Khung hiển thị ảnh (kích thước tính theo pixel, không tự phình ra) -----
        self.left = tk.Frame(root, bg="#222", width=VIEW_W, height=VIEW_H)
        self.left.pack_propagate(False)
        self.left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.view = tk.Label(self.left, bg="#222",
                             text="Mở ảnh hoặc bật webcam để bắt đầu", fg="#ccc")
        self.view.pack(fill=tk.BOTH, expand=True)
        self._last_disp = None
        self.left.bind("<Configure>", lambda e: self._last_disp is not None and self.show(self._last_disp))

        ttk.Label(right, text="Nguồn ảnh", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        ttk.Button(right, text="Mở ảnh...", command=self.open_image).pack(fill=tk.X, pady=2)
        self.cam_btn = ttk.Button(right, text="Bật webcam", command=self.toggle_webcam)
        self.cam_btn.pack(fill=tk.X, pady=2)
        cam_row = ttk.Frame(right)
        cam_row.pack(fill=tk.X)
        ttk.Label(cam_row, text="Camera số:").pack(side=tk.LEFT)
        self.cam_idx = tk.IntVar(value=0)
        ttk.Spinbox(cam_row, from_=0, to=5, width=4, textvariable=self.cam_idx).pack(side=tk.LEFT)

        self.use_feat = tk.BooleanVar(value=True)
        ttk.Checkbutton(right, text="So khớp đặc trưng ORB (cần ảnh mẫu)",
                        variable=self.use_feat).pack(anchor="w", pady=(8, 2))

        ttk.Separator(right).pack(fill=tk.X, pady=8)
        ttk.Label(right, text="Kết quả", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.result_box = tk.Text(right, width=42, height=14, font=("Consolas", 10))
        self.result_box.pack(fill=tk.X)
        self.total_lbl = ttk.Label(right, text="Tổng: 0 đ", font=("Segoe UI", 14, "bold"),
                                   foreground="#0a7")
        self.total_lbl.pack(anchor="w", pady=4)

        ttk.Separator(right).pack(fill=tk.X, pady=8)
        ttk.Label(right, text="Ảnh mẫu (tăng độ chính xác)",
                  font=("Segoe UI", 11, "bold")).pack(anchor="w")
        row = ttk.Frame(right)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="Mệnh giá:").pack(side=tk.LEFT)
        self.denom_var = tk.StringVar(value="500000")
        ttk.Combobox(row, textvariable=self.denom_var, width=10, state="readonly",
                     values=[str(d) for d in R.DENOMINATIONS]).pack(side=tk.LEFT, padx=4)
        ttk.Button(right, text="Lưu tờ tiền hiện tại làm mẫu",
                   command=self.save_template).pack(fill=tk.X, pady=2)
        ttk.Button(right, text="Mở thư mục mẫu", command=self.open_template_folder).pack(fill=tk.X, pady=2)
        ttk.Button(right, text="Nạp lại mẫu", command=self.reload_templates).pack(fill=tk.X, pady=2)
        self.tpl_lbl = ttk.Label(right, text="", wraplength=300, justify=tk.LEFT)
        self.tpl_lbl.pack(anchor="w", pady=4)
        self.update_template_label()

        root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ------------------------------------------------------------------ #
    def update_template_label(self):
        c = self.rec.db.counts()
        if not c:
            txt = "Chưa có ảnh mẫu -> chỉ nhận diện theo màu (tiền polymer 10k-500k)."
        else:
            txt = "Đã có mẫu: " + ", ".join(f"{R.format_vnd(k)}×{v}" for k, v in sorted(c.items()))
        self.tpl_lbl.config(text=txt)

    def show(self, bgr):
        self._last_disp = bgr
        h, w = bgr.shape[:2]
        vw = max(self.left.winfo_width(), 200)
        vh = max(self.left.winfo_height(), 200)
        s = min(vw / w, vh / h)
        disp = cv2.resize(bgr, (max(1, int(w * s)), max(1, int(h * s))))
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.view.config(image=self._photo, text="")

    def show_results(self, results):
        self.result_box.delete("1.0", tk.END)
        for i, r in enumerate(results, 1):
            self.result_box.insert(tk.END, f"Tờ {i}: {r.label}  ({r.confidence * 100:.0f}%)\n")
            top = sorted(r.scores.items(), key=lambda kv: -kv[1])[:3]
            for d, v in top:
                extra = f", ORB {r.inliers.get(d, 0)} điểm" if r.inliers else ""
                self.result_box.insert(tk.END, f"   {R.format_vnd(d):>12}: {v:.2f}{extra}\n")
        total = sum(r.denom for r in results if r.denom)
        self.total_lbl.config(text=f"Tổng: {R.format_vnd(total)}  ({len(results)} tờ)")

    def process(self, img):
        small, res = self.rec.recognize(img, use_features=self.use_feat.get())
        self.last_img, self.last_results = small, res
        self.show(R.draw_results(small, res))
        self.show_results(res)

    # ------------------------------------------------------------------ #
    def open_image(self):
        if self.cap is not None:
            self.toggle_webcam()
        p = filedialog.askopenfilename(
            title="Chọn ảnh tờ tiền",
            filetypes=[("Ảnh", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff"), ("Tất cả", "*.*")])
        if not p:
            return
        img = R.imread_unicode(p)
        if img is None:
            messagebox.showerror("Lỗi", "Không đọc được ảnh.")
            return
        self.process(img)

    def toggle_webcam(self):
        if self.cap is None:
            idx = self.cam_idx.get()
            backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
            cap = cv2.VideoCapture(idx, backend)
            if not cap.isOpened():
                messagebox.showerror("Lỗi", f"Không mở được camera số {idx}.")
                return
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap = cap
            self.cam_btn.config(text="Tắt webcam (giữ khung hình)")
            self.frame_no = 0
            self.webcam_loop()
        else:
            self.cap.release()
            self.cap = None
            self.cam_btn.config(text="Bật webcam")

    def webcam_loop(self):
        if self.cap is None:
            return
        ok, frame = self.cap.read()
        if ok:
            self.frame_no += 1
            if self.frame_no % WEBCAM_EVERY == 0 or not self.last_results:
                small, res = self.rec.recognize(frame, use_features=self.use_feat.get())
                self.last_results = res
                self.show_results(res)
            else:
                small = R.resize_max(frame)
            self.last_img = small
            self.show(R.draw_results(small, self.last_results))
        self.root.after(15, self.webcam_loop)

    # ------------------------------------------------------------------ #
    def save_template(self):
        if self.last_img is None:
            messagebox.showinfo("Thông báo", "Hãy mở ảnh hoặc bật webcam trước.")
            return
        if len(self.last_results) > 1:
            if not messagebox.askyesno("Nhiều tờ tiền",
                                       "Ảnh có nhiều tờ. Chỉ lưu tờ lớn nhất làm mẫu?"):
                return
        note = R.extract_main_note(self.last_img)
        folder = os.path.join(self.rec.db.folder, self.denom_var.get())
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"mau_{time.strftime('%Y%m%d_%H%M%S')}.png")
        R.imwrite_unicode(path, note)
        self.reload_templates()
        messagebox.showinfo("Đã lưu", f"Đã lưu mẫu:\n{path}")

    def reload_templates(self):
        self.rec.db.reload()
        self.update_template_label()

    def open_template_folder(self):
        folder = self.rec.db.folder
        os.makedirs(folder, exist_ok=True)
        if os.name == "nt":
            os.startfile(folder)
        else:
            messagebox.showinfo("Thư mục mẫu", folder)

    def on_close(self):
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista" if os.name == "nt" else "clam")
    except tk.TclError:
        pass
    App(root)
    root.mainloop()
