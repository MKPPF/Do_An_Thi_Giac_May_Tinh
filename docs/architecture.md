# Kiến trúc CrackSpot

## Mục tiêu thiết kế

1. Một implementation tiền xử lý duy nhất cho train, evaluate, CLI và Streamlit.
2. Tách dữ liệu, modeling, inference và reporting để kiểm thử độc lập.
3. Config, manifest, checkpoint và artifact có hash để tái lập.
4. Test bị khóa tới khi lựa chọn checkpoint/threshold hoàn tất.
5. Demo là một tiến trình Streamlit, không thêm API/database ngoài phạm vi.

## Luồng dữ liệu và huấn luyện

```text
Archive SDNET2018
       |
       v
download/checksum/safe extract
       |
       v
audit decode + label/surface + SHA-256 + duplicate report
       |
       v
source_group đã kiểm chứng? --không--> fail full experiment + candidate group map
       |
      có
       v
exact hash có nhãn mâu thuẫn? --có--> conflict report + loại toàn bộ dòng thuộc hash
       |                                  |                         |
      không                               +--> manifest làm sạch <--+
       |                                                           |
       +-----------------------------------------------------------+
                                  |
                                  v
group-aware split 70/15/15 + split_audit + chuỗi manifest hashes
       |
       v
tf.data (train-only augmentation, train-only class weight)
       |
       v
MobileNetV2 E1/E2/E3 -> validation-only selection -> E4 augmentation
       |
       v
E5 validation threshold -> selection_complete.json -> final test một lần
       |
       v
metrics/predictions/plots/Grad-CAM/latency -> report_facts.json
```

## Luồng suy luận

```text
JPG/JPEG/PNG bytes
  -> kiểm tra dung lượng/decode/decompression bomb/EXIF
  -> RGB + bản ảnh gốc không bị ghi đè
  -> resize 224x224 + MobileNetV2 preprocess_input
  -> full .keras model
  -> sigmoid P(Crack)
  -> threshold đã khóa
  -> Grad-CAM target Crack tại out_relu
  -> kết quả CLI/Streamlit
```

## Thành phần

| Khối | Trách nhiệm | Không được làm |
|---|---|---|
| `src/crackspot/data` | audit, manifest, split, `tf.data` | tự random split khi chưa có source group |
| `src/crackspot/modeling` | model, freeze/unfreeze, train, threshold, evaluate, Grad-CAM | đọc test để chọn model/threshold |
| `src/crackspot/inference` | decode/preprocess/service dùng chung | tin extension/MIME; lưu upload mặc định |
| `src/crackspot/reporting` | plot và artifact báo cáo | bịa/điền metric khi thiếu artifact |
| `scripts` | CLI orchestration mỏng | duplicate business logic |
| `app.py` | UI Streamlit tiếng Việt | huấn luyện, DB, auth hay logic preprocess riêng |
| `CrackSpot.ipynb` | bảng điều khiển local/Colab, gọi scripts, hiển thị inventory | chép lại model/data pipeline |

## Hợp đồng model và metadata

Checkpoint bàn giao là full `.keras`. Metadata đi kèm tối thiểu:

- `run_id`, `model_version`;
- `threshold`;
- `input_size: [224,224]`;
- `preprocessing: mobilenet_v2.preprocess_input`;
- `label_mapping: {Non-crack: 0, Crack: 1}`;
- `gradcam_layer: out_relu`;
- `tensorflow_version`;
- `model_sha256`, `manifest_sha256`.

App mặc định đọc `models/crackspot.keras` và `models/crackspot.metadata.json`; có thể override bằng biến môi trường được mô tả trong README.

## Quy tắc nhất quán

- Score luôn là P(Crack), kể cả dự đoán Non-crack.
- Threshold chỉ đến từ metadata/selection lock, không hard-code khác nhau giữa CLI và UI.
- Grad-CAM target score Crack và layer phải được resolve thực tế trong model/nested backbone.
- Validation/test không augmentation.
- BatchNormalization frozen khi fine-tune theo E2-E4.
- Mọi path trong code là tương đối/`pathlib`; config không chứa `D:\`.

## Ranh giới tin cậy và bảo mật

- Upload là input không tin cậy: giới hạn bytes/pixels, decode thật và xử lý lỗi thân thiện.
- Zip dataset là input không tin cậy: safe extract, không path traversal.
- Metadata/checkpoint phải xác minh SHA-256 khi có giá trị công bố.
- Không commit secret, dataset, upload người dùng hay checkpoint lớn ngoài Git LFS/release.
- Artifact smoke mang cờ `NOT_VALID_FOR_REPORT` và nằm ngoài `artifacts/report/`.
