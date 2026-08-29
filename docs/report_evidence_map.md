# Bản đồ bằng chứng cho báo cáo

`artifacts/report/final_bundle_v1/report_facts.json` là nguồn số liệu duy nhất. Mọi bảng/hình trong báo cáo phải ghi run ID, checkpoint hash và manifest hash, hoặc trỏ tới facts chứa các hash đó.

| Nội dung báo cáo | Mã nguồn/config | Bằng chứng/artifact bắt buộc | Quy tắc sử dụng |
|---|---|---|---|
| Nguồn và phân bố dữ liệu | data audit + base config | audit gốc, `conflict_rows.csv`, `conflict_report.json`, `dataset_summary.csv/json` sau curation | không dùng số tham chiếu thay audit; nêu rõ bốn dòng bị loại |
| Chống leakage | split module | manifest audit/làm sạch/split, `split_audit.json`, chuỗi manifest hashes | source group phải verified; không còn exact hash mâu thuẫn |
| Tiền xử lý/augmentation | pipeline/inference | config snapshot, batch before/after | validation/test không aug |
| Kiến trúc MobileNetV2 | model module | model summary, metadata | ghi total/trainable params thật |
| E1-E4 | experiment YAML/train | history, validation metrics, checkpoints | cùng split/hash |
| Chọn E4 | select model | `model_selection.json` | chỉ validation |
| E5 threshold | threshold module | threshold PR/F1 curve, selected threshold | không đọc test |
| Khóa test | final evaluator | `selection_complete.json` | hash phải khớp |
| Kết quả test | evaluate | metrics/predictions/classification report | chỉ final run |
| Confusion matrix | reporting | raw + normalized figures/data | ghi FP/FN |
| Learning curves | reporting | history CSV + plots | không làm mượt che overfit |
| Grad-CAM | gradcam/reporting | grid TP/TN/FP/FN | không gọi mask |
| Ảnh thực tế | real manifest/evaluate | cỡ mẫu, metrics/predictions riêng | không gộp test chuẩn |
| Latency | benchmark | environment + raw timings + summary | warm-up, mean, p50, p95 |
| Demo | app/service | Streamlit verification log/screenshot thật | cùng model/metadata với CLI |
| Hạn chế | analysis | FP/FN, per-surface/domain evidence | phân biệt dữ kiện/suy luận |

## Quy tắc đặt tên hình/bảng đề xuất

- `fig_dataset_distribution.*`
- `fig_training_curves_e1...e4.*`
- `fig_confusion_matrix_raw.*`, `fig_confusion_matrix_normalized.*`
- `fig_threshold_curves_e5.*`
- `fig_gradcam_tp_tn_fp_fn.*`
- `table_experiment_comparison.*`
- `table_latency_benchmark.*`

Không tạo placeholder có hình giả trong `artifacts/report/`. Nếu thiếu artifact, handoff phải ghi **CHƯA CÓ BẰNG CHỨNG**.

## Audit trước khi chèn vào báo cáo

- [ ] Con số tồn tại trong `report_facts.json`.
- [ ] Hash checkpoint/manifest khớp selection lock.
- [ ] Metric ghi rõ split và threshold.
- [ ] Hình có caption, nguồn và run ID.
- [ ] Accuracy mục tiêu không bị viết thành kết quả.
- [ ] Smoke/synthetic không xuất hiện trong report assets.
- [ ] Conflict report nối đúng canonical hash manifest audit và số liệu trước/sau curation khớp.
- [ ] Kết quả ảnh thực tế ghi cỡ mẫu và tách riêng.
- [ ] Grad-CAM được gọi là heatmap/vùng chú ý.
