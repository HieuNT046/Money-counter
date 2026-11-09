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

**Đánh giá nhận diện nhiều tờ / tờ bị đè:**

```
python tao_bo_test.py --out bo_test --n 150 --seed 1   # sinh ảnh ghép có nhãn chuẩn
python danh_gia.py bo_test                              # cách mới (SIFT + RANSAC lặp)
python danh_gia.py bo_test --method classic             # cách cũ để so sánh
python danh_gia.py bo_test --save-errors loi/           # lưu ảnh các trường hợp sai
```

Với ảnh chụp thật, tạo `labels.json` chỉ cần ghi mệnh giá từng tờ (không cần vẽ khung):
`[{"file": "IMG_01.jpg", "notes": [{"denom": 500000}, {"denom": 20000}]}]`.

## Nhận diện nhiều tờ, tờ bị đè

Cách tách nền rồi tìm hình chữ nhật thất bại khi các tờ chồng lên nhau: chúng dính thành một khối không còn tỉ lệ của tờ tiền. Phần mềm tìm **trực tiếp từng tờ bằng điểm đặc trưng cục bộ**, nên chỉ cần phần còn nhìn thấy có đủ điểm khớp:

1. **SIFT** trên toàn ảnh và trên mọi ảnh mẫu (nắn về 1000×450). Chỉ mục FLANN (k-d tree).
2. **Ratio test theo lớp** (Lowe 2004): giữ một cặp khớp chỉ khi láng giềng gần nhất gần hơn hẳn láng giềng gần nhất thuộc *mệnh giá khác*. Chi tiết chung của mọi tờ (chân dung, quốc huy, quốc hiệu) không phân biệt được nên tự bị loại.
3. **RANSAC lặp** (sequential RANSAC, ước lượng MAGSAC): lấy homography có nhiều inlier nhất trong mọi ảnh mẫu → nhận một tờ → xóa các điểm đã dùng → lặp lại. Chiếu 4 góc ảnh mẫu qua homography được khung đầy đủ của tờ, kể cả phần bị che.
4. **Kiểm tra hình học**: không lật gương (det > 0), tứ giác lồi, tỉ lệ cạnh 1.5–3.2, inlier phủ đủ rộng (dải hẹp thì cần nhiều inlier hơn); loại phát hiện trùng khi 4 góc gần như trùng nhau.
5. Vùng tìm được bằng tách nền mà không trùng tờ nào ở trên được phân loại bằng màu (tờ chưa có ảnh mẫu).

Kết quả trên bộ ảnh ghép `tao_bo_test.py --seed 2` (150 ảnh, 483 tờ, 2–6 tờ/ảnh; tham số được chỉnh trên bộ `--seed 1`):

| | Cách cũ | Cách mới |
|---|---|---|
| Tờ nhận đúng (vị trí + mệnh giá) | 0.6% | **96.9%** |
| Dự đoán đúng (precision) | 33% | **100%** |
| Ảnh đúng tổng tiền | 0% | **92.0%** |
| Tờ chỉ nhìn thấy < 40% | 0% | 82.3% |
| Tờ nhìn thấy 40–70% / ≥ 70% | 0% | 99.0% / 100% |

Lưu ý: tờ tiền trong ảnh ghép lấy từ chính ảnh mẫu nên kết quả lạc quan hơn ảnh chụp thật; cần bổ sung ảnh chụp thật để đánh giá cuối cùng. Thời gian ~1 s/ảnh (SIFT); đặt `FEATURE_KIND = "orb"` để nhanh hơn.

## Nguyên lý (cách cổ điển, dùng khi tắt so khớp đặc trưng)

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
- Ảnh mẫu được lưu đệm vào `templates/_cache.pkl` và `templates/_cache_sift.pkl`. Lần chạy sau chỉ tính lại ảnh mới hoặc ảnh đã sửa. Nếu cần tính lại toàn bộ thì chạy `python recognizer.py --rebuild` (hoặc xóa các file cache).
- Có thể chỉnh các ngưỡng ở đầu `recognizer.py`: `COLOR_PROTOTYPES`, `W_FEATURE`, `W_COLOR`, `MIN_CONFIDENCE`; phần nhận diện nhiều tờ: `FEATURE_KIND`, `CLASS_RATIO`, `MIN_INLIERS`, `TPL_FEATURES`.

## Cấu trúc

```
nhan_dien_tien/
├── main.py           # giao diện Tkinter (ảnh + webcam, nhận diện ở luồng riêng)
├── recognizer.py     # thuật toán nhận diện + dòng lệnh
├── tao_bo_test.py    # sinh bộ ảnh test tiền bị đè có nhãn chuẩn
├── danh_gia.py       # đo độ chính xác nhận diện nhiều tờ
├── requirements.txt
└── templates/        # ảnh mẫu: templates/500000/*.jpg ...
```
