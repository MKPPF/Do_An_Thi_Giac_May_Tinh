# Hướng dẫn sử dụng CrackSpot

## Chuẩn bị checkpoint

Demo cần:

```text
models/crackspot.keras
models/crackspot.metadata.json
```

Hoặc dùng biến môi trường:

```powershell
$env:CRACKSPOT_MODEL_PATH = "D:\models\crackspot.keras"
$env:CRACKSPOT_METADATA_PATH = "D:\models\crackspot.metadata.json"
```

Trước khi chạy:

```powershell
Get-FileHash .\models\crackspot.keras -Algorithm SHA256
```

Hash phải khớp metadata/release. Nếu thiếu/hỏng model, UI phải thông báo rõ thay vì crash.

## Khởi động

```powershell
cd D:\Đồ án\CrackSpot
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

## Dự đoán trên giao diện

1. Chọn hoặc kéo-thả một ảnh JPG, JPEG hay PNG.
2. Kiểm tra preview và không dùng ảnh có thông tin nhạy cảm nếu chạy cloud.
3. Nhấn **Phát hiện vết nứt**.
4. Đọc:
   - nhãn Có vết nứt/Không phát hiện vết nứt;
   - xác suất Crack;
   - threshold;
   - run/model version;
   - thời gian xử lý;
   - ảnh gốc và Grad-CAM overlay.

Mặc định file tối đa 10 MB và 25 megapixel. Hệ thống decode bytes thật, tôn trọng EXIF orientation, chuyển grayscale/RGBA hợp lệ về RGB và từ chối file hỏng/giả extension/quá lớn.

## Hiểu kết quả

- `P(Crack) >= threshold`: Có vết nứt.
- `P(Crack) < threshold`: Không phát hiện vết nứt.
- “Không phát hiện” không chứng minh bề mặt an toàn.
- Grad-CAM là vùng kích hoạt score Crack, không phải mask pixel. Với ảnh Non-crack, heatmap vẫn cho biết vùng có thể làm tăng score Crack.

Không dùng kết quả để đo vết nứt, xếp hạng nguy hiểm hoặc bỏ qua kiểm tra chuyên môn.

## CLI

```powershell
python scripts/predict.py .\samples\wall.jpg
```

Xem tham số thật:

```powershell
python scripts/predict.py --help
```

CLI và UI phải trả cùng xác suất trong sai số float trên cùng ảnh/model/metadata.

## Quyền riêng tư

Thiết kế demo xử lý upload trong bộ nhớ và không lưu lịch sử. Tuy nhiên, khi deploy trên cloud, bytes vẫn được truyền tới máy chủ. Chỉ upload ảnh bạn có quyền xử lý; tránh người, biển số, địa chỉ hoặc dữ liệu nhạy cảm.

## Xử lý sự cố

- **File bị từ chối:** thử mở lại bằng trình xem ảnh và xuất JPG/PNG chuẩn; không chỉ đổi extension.
- **Ảnh quá lớn:** giảm độ phân giải trước khi upload; không nới limit tùy ý trên public demo.
- **Thiếu model/metadata:** làm theo mục Chuẩn bị checkpoint.
- **Hash sai:** không chạy model; tải lại từ nguồn bàn giao đáng tin cậy.
- **Heatmap trống/lạ:** vẫn giữ dự đoán nhưng ghi nhận vào phân tích lỗi; không chỉnh heatmap để đẹp.
- **Kết quả bất thường:** chuyển cho kỹ sư xây dựng và không dùng CrackSpot làm căn cứ duy nhất.
