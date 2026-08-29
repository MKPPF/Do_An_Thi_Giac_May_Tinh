# Protocol thực nghiệm và khóa test

## Nguyên tắc bất biến

- Seed `42` cho Python, NumPy, TensorFlow; bật deterministic operations khi hỗ trợ và ghi cảnh báo phần không deterministic.
- Một manifest/split đã khóa cho E1-E5.
- Split `70/15/15` theo `source_group` đã xác minh; không giao path/group/exact hash.
- Chỉ train có augmentation; class weight chỉ tính từ train.
- Không thay split sau khi xem test.
- Không dùng test để chọn kiến trúc, checkpoint, threshold hoặc dừng epoch.
- Nếu Accuracy thật dưới `0,92`, giữ nguyên và phân tích; không loại mẫu hay chạy lại để chọn số đẹp.

## Cổng dữ liệu

Full experiment chỉ hợp lệ nếu:

1. archive/dataset thật đã audit và checksum native khớp;
2. quy tắc `source_group` được xác minh bằng archive/README hoặc `group_map.csv`;
3. mọi exact hash có nhãn mâu thuẫn được xử lý **trước split** bằng conflict report bất biến;
4. manifest audit gốc được giữ nguyên; manifest làm sạch loại toàn bộ dòng thuộc hash mâu thuẫn, công bố chênh lệch và nối hash provenance về manifest gốc;
5. manifest làm sạch và split có SHA-256;
6. `split_audit.json` pass path/group/exact hash;
7. config và manifest hash được snapshot.

Nếu thiếu source group, chỉ được smoke với nhãn `NOT_VALID_FOR_REPORT`.

### Trạng thái audit SDNET2018 hiện tại

- Native MD5 đã xác minh: `677411e784f194422c90f52d9ed0d7c6`.
- Audit: 56.092 ảnh hợp lệ, 0 lỗi; 8.484 Crack và 47.608 Non-crack.
- Nhóm nguồn đã xác minh: 230 nhóm, D=54, P=104, W=72.
- Canonical SHA-256 manifest audit: `3b2acbaddb5431726c08bc4562d3f515210f0827fbbc057e1c08c9dd70369f5f`.
- Curation đã khóa tại `data/manifests/pre_split_curation_v1/`: loại 4 dòng thuộc 2 hash mâu thuẫn, giữ 56.088 dòng/230 nhóm, không còn hash xung đột; canonical hash sạch `2ffb560a49aadfe475129ca56b0ae90741c37913a4657dd60cd51839cea880c7`.
- Locked `split_v1` đã xác minh: train/validation/test = 39.014/8.540/8.534 ảnh, 160/35/35 nhóm; canonical SHA-256 `b38302d62547fb20f264b811f38f7031cf1b8d4c8a21f9db15d4dc39699a3da7`; zero path/group/hash overlap và đủ sáu strata mỗi split.
- Full byte-integrity scan pass 56.088 ảnh/530.749.950 byte với fingerprint `7bed031907031ba302d42fc1349b522d11300ff4f54a4740bae56a3d92caa4fd`.

Không suy đoán nhãn đúng, không giữ tùy ý một phía và không xóa file khỏi archive gốc. Loại cả hai phía của từng hash mâu thuẫn khỏi manifest tiền split là quyết định chất lượng dữ liệu có thể tái lập, không phải test cherry-picking.

CLI và đường dẫn artifact chuẩn cho lần curation đầu tiên:

```powershell
python scripts/curate_manifest.py data/manifests/audit_manifest.csv `
  --output-dir data/manifests/pre_split_curation_v1
```

Thư mục output đã khóa chứa `pre_split_manifest.csv`, `conflict_rows.csv` và `conflict_report.json`; chạy lại cùng output bị từ chối. Splitter chỉ nhận `pre_split_manifest.csv` sau khi các hash trong report đã được kiểm tra.

## E1-E5

| ID | Training | Augmentation | Lựa chọn |
|---|---|---|---|
| E1 | head LR 1e-3, backbone frozen | không | checkpoint val_loss |
| E2 | head rồi unfreeze `block_14_expand`, LR 1e-4, BN frozen | không | checkpoint val_loss |
| E3 | head rồi unfreeze `block_10_expand`, LR 1e-5, BN frozen | không | checkpoint val_loss; kiểm tra overfit |
| E4 | kế thừa E2/E3 thắng validation | có | quyết định ghi `model_selection.json` |
| E5 | không train; checkpoint E4 | không áp dụng | tune threshold trên validation |

E4 không được mặc định E2 hay E3 thắng. Trước run phải resolve layer/LR từ `model_selection.json` vào config snapshot.

## Callback và checkpoint

- head tối đa 20 epoch; fine-tune tối đa 30 epoch;
- `EarlyStopping(val_loss, patience=5, restore_best_weights=True)`;
- `ModelCheckpoint(val_loss, save_best_only=True, full .keras)`;
- `ReduceLROnPlateau(val_loss, factor=0.2, patience=2, min_lr=1e-7)`;
- lưu số layer/parameter trainable và xác nhận BatchNormalization frozen.

## Threshold E5

1. Dùng predictions validation của checkpoint E4 tốt nhất.
2. Tối đa F1 lớp Crack.
3. Nếu hòa, chọn Recall Crack cao hơn.
4. Nếu vẫn hòa, chọn threshold gần `0.5` hơn; tie cuối dùng thứ tự xác định trong config.
5. Khóa threshold trước test.

Tuner xét mọi xác suất validation duy nhất cùng các mốc `0`, `0.5`, `1`; đây là tập đầy đủ các điểm có thể đổi dự đoán nhị phân, tránh bỏ lỡ nghiệm do lưới bước cố định.

Threshold tuner phải từ chối test manifest/predictions.

## Selection lock

Trước final test phải tạo `selection_complete.json` tối thiểu gồm:

```json
{
  "checkpoint_sha256": "...",
  "config_sha256": "...",
  "manifest_sha256": "...",
  "threshold": 0.5,
  "selection_split": "validation",
  "created_at_utc": "..."
}
```

Giá trị trên chỉ minh họa schema. Final evaluator phải từ chối khi thiếu lock hoặc hash không khớp.

## Final evaluation

- Mỗi checkpoint chỉ được đánh giá test một lần trong giai đoạn chính thức.
- E1-E4 được báo cáo ở threshold `0.5`.
- Checkpoint E4 tốt nhất báo cáo thêm ở threshold E5.
- Nêu rõ Accuracy `>=0,92` đạt ở threshold nào.
- Ảnh thực tế tự chụp báo cáo riêng, ghi cỡ mẫu; không gộp vào test chuẩn.

Metric bắt buộc:

- Accuracy;
- Precision, Recall, F1 của Crack;
- macro average khi hữu ích;
- confusion matrix số đếm và chuẩn hóa;
- FP, FN;
- total/trainable parameters;
- thời gian huấn luyện;
- latency warm-up, mean, median/p50, p95;
- train/validation loss và Accuracy;
- Grad-CAM TP/TN/FP/FN.

## Artifact mỗi run

- config snapshot/hash, manifest/split hash;
- environment, `pip freeze`, Git commit;
- seed/deterministic;
- log, `history.csv`;
- checkpoint/hash;
- validation metrics/predictions;
- classification report/confusion matrix/curves;
- parameter count, duration, latency;
- FP/FN và ảnh phân tích theo quy tắc tái lập.

`metrics_test.json` và `predictions_test.csv` chỉ có sau final evaluation. Không ghi đè run cũ.

## Resume an toàn

- `--resume` luôn đi cùng đúng `--run-id` đã tồn tại; fresh run vẫn từ chối thư mục trùng.
- Trước khi gọi `fit`, hệ thống xác minh nguyên giá trị config, hash semantic/file của snapshot và canonical manifest hash.
- Backup, CSV journal và completion marker tách theo phase head/fine-tune; marker khóa cả checkpoint và log hash.
- Phase đã hoàn tất được bỏ qua; phase bị ngắt nối tiếp log và phục hồi trạng thái callback. Epoch trùng nhưng khác nội dung bị từ chối.
- Run có `training_complete.json` hoặc `run_summary.json` không được resume. Final evidence chỉ ghi sau khi toàn bộ training/validation hoàn tất.

## Tính hợp lệ của kết quả

| Nhãn | Ý nghĩa |
|---|---|
| `NOT_VALID_FOR_REPORT` | synthetic/tiny, split không group, smoke hoặc pipeline debug |
| validation result | dùng lựa chọn nhưng không phải kết quả test |
| final test result | có selection lock/hash khớp và log final evaluation |

Chỉ `report_facts.json` tổng hợp từ final artifacts được phép cung cấp số liệu cho báo cáo/slide.
