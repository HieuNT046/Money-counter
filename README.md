# Nhận diện mệnh giá tiền Việt Nam (không dùng AI)

Phần mềm Python nhận diện mệnh giá tiền Việt Nam bằng **xử lý ảnh cổ điển** (OpenCV): không mạng nơ-ron, không học máy.

## Cài đặt

```
pip install -r requirements.txt
```

Cần Python 3.8 trở lên (trên Windows, Tkinter đã có sẵn).

## Cách dùng

**Giao diện:**

```
python main.py
```

- **Mở ảnh...**: chọn ảnh có một hoặc nhiều tờ tiền. Phần mềm khoanh từng tờ, ghi mệnh giá và tính tổng.
- **Bật webcam**: nhận diện trực tiếp. Nếu máy có nhiều camera, đổi "Camera số".
- **Lưu tờ tiền hiện tại làm mẫu**: chọn đúng mệnh giá rồi bấm để lưu tờ đang hiển thị vào `templates/<mệnh giá>/`.

**Dòng lệnh:**

```
python recognizer.py anh1.jpg anh2.jpg        # in kết quả, lưu anh1_ketqua.jpg
python recognizer.py --eval thu_muc_kiem_tra  # đo độ chính xác
```

Thư mục kiểm tra có dạng `thu_muc_kiem_tra/500000/*.jpg`, `thu_muc_kiem_tra/20000/*.jpg`...

## Nguyên lý

1. **Tiền xử lý**: thu nhỏ ảnh, cân bằng trắng theo các vùng sáng trung tính để giảm ảnh hưởng của màu đèn.
2. **Định vị tờ tiền**: tách vật thể khỏi nền (khác màu nền trong không gian Lab + cạnh Canny), giữ các vùng chữ nhật có tỉ lệ dài/rộng 1.7–2.9, rồi nắn phối cảnh về 640×288.
3. **Màu sắc**: histogram Hue (HSV) có trọng số độ bão hòa, so bằng phép giao histogram với màu chủ đạo:

   | Mệnh giá | Màu chủ đạo |
   |---|---|
   | 10.000 | vàng nâu |
   | 20.000 | xanh dương |
   | 50.000 | hồng tím |
   | 100.000 | xanh lá |
   | 200.000 | đỏ cam |
   | 500.000 | xanh lơ |

4. **Đặc trưng ORB**: so khớp điểm đặc trưng với ảnh mẫu (tỉ lệ Lowe 0.75), lọc bằng RANSAC homography, đếm số điểm khớp đúng (inlier).
5. **Kết hợp**: `điểm = 0.65 × ORB + 0.35 × màu`. Nếu chưa có ảnh mẫu thì chỉ dùng màu.

## Để nhận diện chính xác

- **Nên thêm ảnh mẫu**, cả mặt trước và mặt sau, mỗi mệnh giá 2–4 ảnh, chụp bằng chính camera sẽ dùng. Khi có mẫu, màu chuẩn cũng được học lại từ ảnh mẫu thay cho bảng màu mặc định.
- Đặt tờ tiền trên nền trơn, khác màu tiền (ví dụ bàn trắng hoặc xám), ánh sáng đều, không bị che.
- Tiền giấy cũ (1.000, 2.000, 5.000) không có màu mặc định, cần ảnh mẫu thì mới nhận diện được.
- Ảnh mẫu được lưu đệm vào `templates/_cache.pkl`. Lần chạy sau chỉ tính lại ảnh mới hoặc ảnh đã sửa. Nếu cần tính lại toàn bộ thì chạy `python recognizer.py --rebuild` (hoặc xóa file cache).
- Có thể chỉnh các ngưỡng ở đầu `recognizer.py`: `COLOR_PROTOTYPES`, `W_FEATURE`, `W_COLOR`, `MIN_CONFIDENCE`.

## Cấu trúc

```
nhan_dien_tien/
├── main.py           # giao diện Tkinter (ảnh + webcam)
├── recognizer.py     # thuật toán nhận diện + dòng lệnh
├── requirements.txt
└── templates/        # ảnh mẫu: templates/500000/*.jpg ...
```
