# Kịch bản bảo vệ demo 5-7 phút

## Chuẩn bị trước buổi bảo vệ

- Xác minh checkpoint/metadata SHA-256.
- Chạy `pytest -q` và smoke Streamlit; lưu log.
- Chọn trước 1 ảnh Crack, 1 Non-crack, 1 FP hoặc FN thật từ artifact và 1 file hỏng.
- Mở `report_facts.json`; không ghi nhớ số liệu từ bản nháp.
- Có phương án offline: môi trường cục bộ, model và bốn ảnh demo; không phụ thuộc Internet.

## Kịch bản

### 0:00-0:45 - Bài toán và giới hạn

“CrackSpot phân loại ảnh bề mặt thành Crack và Non-crack bằng MobileNetV2 transfer learning. Hệ thống hỗ trợ khảo sát sơ bộ; Grad-CAM chỉ là vùng mô hình chú ý, không phải mask và không thay đánh giá kỹ sư.”

### 0:45-1:30 - Dữ liệu và protocol

Trình bày SDNET2018, split theo source group 70/15/15, seed 42 và test-lock. Nêu cỡ mẫu/hash theo artifact thật. Nhấn mạnh ảnh tự chụp được đánh giá riêng.

### 1:30-2:30 - Ảnh Crack

Upload ảnh Crack, nhấn nút, chỉ nhãn, P(Crack), threshold, version, latency và Grad-CAM. Giải thích heatmap có/không tập trung vào vùng nứt nhìn thấy.

### 2:30-3:15 - Ảnh Non-crack

Lặp lại. Giải thích heatmap vẫn target score Crack và “Không phát hiện” không đồng nghĩa an toàn tuyệt đối.

### 3:15-3:45 - Xử lý lỗi

Upload file hỏng/giả extension; cho thấy hệ thống từ chối thân thiện, không crash và không lưu ảnh.

### 3:45-4:45 - Kết quả định lượng

Đọc **chính xác** từ `report_facts.json`: Accuracy, Precision/Recall/F1 Crack, FP/FN, threshold, confusion matrix và latency p95. Nêu Accuracy có đạt 0,92 hay không; không né kết quả dưới mục tiêu.

### 4:45-5:45 - Một lỗi thật

Trình bày một FP/FN thật: ảnh gốc, xác suất, Grad-CAM, nguyên nhân có thể như bóng/mối nối/độ tương phản/domain shift. Phân biệt quan sát với kết luận đã chứng minh.

### 5:45-6:30 - Kết luận

Tóm tắt đóng góp: pipeline tái lập, MobileNetV2, protocol khóa test, demo và bộ bằng chứng. Nêu hạn chế nhãn toàn ảnh, domain shift, heatmap thô và hướng segmentation chỉ khi có mask thật.

## Câu hỏi thường gặp

**Tại sao không object detection/segmentation?**

SDNET2018 dùng trong phạm vi này có nhãn toàn ảnh, không có box/mask được kiểm chứng. Grad-CAM chỉ hỗ trợ giải thích định tính.

**Tại sao dùng MobileNetV2?**

Kiến trúc nhẹ, phù hợp transfer learning và demo trong thời gian/tài nguyên đồ án.

**Accuracy cao có đủ không?**

Không. Dataset mất cân bằng nên phải đọc Recall/F1 Crack, FP/FN và confusion matrix.

**Có tune trên test không?**

Không. E2/E3/E4 chọn bằng validation; E5 tune threshold trên validation; test chỉ mở sau selection lock.

**Grad-CAM có khoanh vết nứt chính xác không?**

Không. Nó cho biết vùng ảnh ảnh hưởng tới score Crack ở độ phân giải thô.

**Kết quả thực tế có dùng được ngay không?**

Chỉ cho khảo sát sơ bộ. Domain shift cần đánh giá thêm và mọi quyết định an toàn cần kỹ sư.

**Nếu Accuracy chưa đạt 0,92?**

Báo đúng kết quả, phân tích dữ liệu/cấu hình/domain shift; không thay split hoặc loại mẫu test.
