# Model Card - CrackSpot MobileNetV2

## Trạng thái

**CHƯA SẴN SÀNG NỘP: chưa có model hoặc metric E1-E5 chính thức.**

Mục tiêu `Accuracy >= 0,92` là tiêu chí của đề tài, không phải kết quả. Chỉ cập nhật bảng metric bên dưới từ `artifacts/report/final_bundle_v1/report_facts.json` sau final evaluation đã khóa. Không chép số liệu từ smoke, validation hay nghiên cứu khác.

Archive SDNET2018 native và audit gốc đã được xác minh (56.092 ảnh hợp lệ; 8.484 Crack; 47.608 Non-crack; 230 `source_group`, D=54/P=104/W=72). Hai exact hash gồm bốn file mang nhãn mâu thuẫn đã được loại minh bạch trước split. Locked split chính thức còn 56.088 dòng, canonical SHA-256 `b38302d62547fb20f264b811f38f7031cf1b8d4c8a21f9db15d4dc39699a3da7`, không giao path/group/hash và đã pass full byte-integrity scan. Chưa được công bố metric E1-E5 cho đến khi có run chính thức từ Git commit sạch và final-test artifacts hợp lệ.

## Mô tả mô hình

- Nhiệm vụ: phân loại ảnh nhị phân.
- Label: `Non-crack = 0`, `Crack = 1`.
- Output: một sigmoid, biểu diễn xác suất lớp Crack.
- Backbone: `tf.keras.applications.MobileNetV2`, `weights="imagenet"`, `include_top=False`, input `(224,224,3)`.
- Head: Global Average Pooling -> Dropout 0.3 -> Dense 1 sigmoid.
- Loss: Binary Cross-Entropy.
- Optimizer: Adam.
- Checkpoint bàn giao: full model định dạng `.keras` cùng metadata JSON.

E1-E4 dùng threshold `0.5` khi so sánh protocol. E5 không phải mô hình mới: checkpoint tốt nhất của E4 được giữ nguyên, threshold được tối ưu **chỉ trên validation** để tối đa F1 Crack; khi hòa, ưu tiên Recall cao hơn rồi threshold gần 0.5 hơn.

## Mục đích sử dụng

- hỗ trợ sàng lọc sơ bộ ảnh bề mặt tường, đường, bê tông hoặc cầu;
- minh họa transfer learning và khả năng giải thích định tính bằng Grad-CAM;
- phục vụ nghiên cứu/đào tạo trong phạm vi đồ án.

## Không được sử dụng để

- thay kết luận của kỹ sư xây dựng hoặc kiểm định công trình;
- đo chiều dài, chiều rộng, độ sâu hoặc mức nguy hiểm của vết nứt;
- đề xuất sửa chữa/bảo trì;
- coi Grad-CAM là mask segmentation hay bounding box chính xác;
- ra quyết định an toàn tự động trong môi trường rủi ro cao;
- suy diễn độ tin cậy ngoài miền dữ liệu mà không kiểm chứng.

## Tiền xử lý

Một implementation phải được dùng chung cho train/evaluate/CLI/Streamlit:

1. xác minh bytes ảnh và decode thật;
2. tôn trọng EXIF orientation;
3. giới hạn mặc định 10 MB và 25 megapixel;
4. chuyển mode hợp lệ sang RGB;
5. resize `224 x 224`;
6. gọi `tf.keras.applications.mobilenet_v2.preprocess_input` (`[0,255] -> [-1,1]`).

Metadata bắt buộc đi cùng model:

```json
{
  "run_id": "<từ artifact thật>",
  "model_version": "<từ artifact thật>",
  "threshold": "<float đã khóa>",
  "input_size": [224, 224],
  "preprocessing": "mobilenet_v2.preprocess_input",
  "label_mapping": {"Non-crack": 0, "Crack": 1},
  "gradcam_layer": "out_relu",
  "model_sha256": "<sha256>",
  "manifest_sha256": "<sha256>",
  "tensorflow_version": "2.19.0"
}
```

Đây là schema minh họa, không phải metadata hợp lệ để suy luận. Placeholder phải được thay bằng artifact thật.

## Dữ liệu và protocol

- Nguồn chính: SDNET2018, CC BY 4.0; xem [`DATA_CARD.md`](DATA_CARD.md).
- Audit native: MD5 `677411e784f194422c90f52d9ed0d7c6`; 56.092 ảnh RGB JPEG 256 x 256, 0 file lỗi.
- `source_group`: tiền tố tên file trước dấu `-`, đã kiểm chứng trên toàn bộ 230 nhóm (D=54, P=104, W=72).
- Curation đã khóa: loại cả hai phía của mỗi exact hash có nhãn mâu thuẫn trước split, lưu conflict report và hash nối về manifest audit gốc; không tự sửa nhãn.
- Locked split: seed 42, theo `source_group`; train/validation/test = 39.014/8.540/8.534 ảnh, canonical SHA-256 `b38302d...99a3da7`. Chưa có model chính thức.
- Class weight: balanced, tính chỉ từ train, dùng nhất quán E1-E4.
- Test chỉ mở sau khi khóa checkpoint/config/manifest/threshold trong `selection_complete.json`.
- Ảnh thực tế tự chụp đánh giá riêng và công bố cỡ mẫu.

## Kết quả định lượng

Chỉ điền từ final artifacts:

| Chỉ số | Kết quả | Bằng chứng |
|---|---:|---|
| Threshold chính thức | Chưa đo | `selection_complete.json` |
| Accuracy test | Chưa đo | `metrics_test.json` |
| Precision Crack | Chưa đo | `metrics_test.json` |
| Recall Crack | Chưa đo | `metrics_test.json` |
| F1 Crack | Chưa đo | `metrics_test.json` |
| Macro F1 | Chưa đo | `metrics_test.json` |
| FP / FN | Chưa đo | confusion matrix / predictions |
| Tổng/trainable parameters | Chưa đo | run metadata |

Mục tiêu Accuracy `>= 0,92` đạt hay không phải ghi theo kết quả thật và nêu rõ threshold `0.5` hay E5.

## Hiệu năng suy luận

Mục tiêu: tối đa 5 giây/ảnh trên môi trường được công bố. Benchmark phải có warm-up, số lần lặp, mean, median/p50, p95, đồng thời ghi CPU/GPU, RAM/VRAM, OS, Python, TensorFlow và batch size.

| Môi trường | Warm-up/lặp | Mean | p50 | p95 | Kết luận <=5 s |
|---|---:|---:|---:|---:|---|
| Chưa benchmark | - | - | - | - | Chưa xác minh |

Không dùng thời gian hiển thị UI hoặc một lần chạy duy nhất làm latency chính thức.

## Grad-CAM

- Target: score/xác suất lớp `Crack`.
- Layer mặc định: `out_relu`, phải resolve đúng cả khi backbone là nested model.
- Heatmap chuẩn hóa `[0,1]`, resize về ảnh gốc và overlay mà không ghi đè ảnh nguồn.
- Với dự đoán Non-crack, heatmap vẫn thể hiện vùng kích hoạt score Crack, không phải mask Non-crack.
- Đánh giá định tính trên TP, TN, FP, FN; không suy diễn độ chính xác định vị.

## Giới hạn và rủi ro

- Dataset mất cân bằng; threshold có thể tạo trade-off Recall/Precision.
- Domain shift do camera, góc chụp, ánh sáng, vật liệu và thời tiết.
- Bóng, mối nối, cạnh, vết bẩn và bề mặt sần có thể gây lỗi.
- Feature map cuối của MobileNetV2 có độ phân giải thô; heatmap mất chi tiết vết nứt mảnh.
- Xác suất sigmoid không mặc nhiên là xác suất đã calibration.
- Nhãn toàn ảnh không cho phép xác nhận độ chính xác theo pixel.

## Tái lập và provenance

Model chính thức chỉ hợp lệ khi có:

- file `.keras` và SHA-256 khớp metadata;
- config snapshot/hash;
- manifest/split hash;
- `selection_complete.json`;
- `environment.json` và `pip freeze`;
- predictions/metrics/confusion matrix test;
- report facts và verification log.

Đặt model tại `models/crackspot.keras` và metadata tại `models/crackspot.metadata.json`, hoặc cấu hình bằng `CRACKSPOT_MODEL_PATH`/`CRACKSPOT_METADATA_PATH`. Xem [`models/README.md`](models/README.md).

## Cảnh báo an toàn

Kết quả chỉ hỗ trợ khảo sát sơ bộ, không thay thế đánh giá của kỹ sư xây dựng và không kết luận mức độ nguy hiểm. Khi mô hình không chắc chắn, ảnh ngoài miền hoặc Grad-CAM chú ý bất thường, phải chuyển cho người có chuyên môn thay vì tự động hành động.
