# Checklist đóng gói và nộp CrackSpot

Chỉ đánh dấu khi có bằng chứng. Nếu còn ô cốt lõi chưa đạt, trạng thái là **CHƯA SẴN SÀNG NỘP**.

## Gate A - Thiết kế

- [x] Traceability cập nhật tới module/test/artifact thực tế.
- [x] Architecture/config/artifact schema được chốt.
- [ ] Protocol test-lock được nhóm/GVHD hiểu.

## Gate B - Code-complete

- [x] Không còn TODO/mock trong logic chính.
- [x] `ruff check .` pass.
- [x] `ruff format --check .` pass.
- [x] `pytest -q` pass offline: 260 tests.
- [x] Tiny smoke train->checkpoint->evaluate->Grad-CAM->predict pass.
- [x] Smoke gắn `NOT_VALID_FOR_REPORT`.
- [x] Streamlit import/start/health smoke pass.
- [ ] Log verification có thời gian/commit/môi trường.

## Gate C - Data-ready

- [x] SDNET2018 native tải từ nguồn; MD5 `677411e784f194422c90f52d9ed0d7c6` khớp; 56.092 ảnh audit hợp lệ, 0 lỗi.
- [x] Phân bố audit đã ghi: 8.484 Crack, 47.608 Non-crack; D/P/W = 13.620/24.334/18.138.
- [x] Source-group rule (tiền tố trước `-`) đã kiểm chứng: 230 nhóm, D/P/W = 54/104/72.
- [x] `pre_split_curation_v1/conflict_rows.csv` và `conflict_report.json` đã khóa cho hai exact hash/bốn file có nhãn mâu thuẫn.
- [x] `pre_split_manifest.csv` đã loại đủ bốn dòng mâu thuẫn, giữ manifest audit gốc và hash provenance; 56.088 dòng, canonical SHA-256 `2ffb560a...880c7`.
- [x] Validation official bundle cho seed/tỷ lệ/strata/chuỗi hash đã được cài đặt và kiểm thử.
- [x] Manifest/split 70/15/15 seed42 đã khóa: 39.014/8.540/8.534 ảnh, 160/35/35 nhóm.
- [x] `split_audit.json` pass path/group/hash và đủ sáu strata ở mỗi split.
- [x] Canonical manifest SHA-256 đã lưu: `b38302d...99a3da7`.
- [ ] Ảnh thực tế + manifest/nhãn có thật; không ảnh giả.

Phần SDNET2018 của Gate C đã đạt và có bằng chứng tại `artifacts/verification/gate_c_split_v1.json`. Gate C tổng vẫn còn ảnh thực tế tự chụp; không chạy E1-E5 chính thức trước khi có Git HEAD sạch. Bốn mẫu mâu thuẫn đã được loại trước split, không dựa trên test.

## Gate D - Experiment-ready

- [ ] E1-E3 cùng split/config snapshot, có log/checkpoint/artifact.
- [ ] E4 selection chỉ validation và có `model_selection.json`.
- [ ] E4 augmentation chạy với cấu hình thắng đã snapshot.
- [ ] E5 threshold chỉ validation và tie-break đúng.
- [ ] `selection_complete.json` hash khớp.
- [ ] Final test chạy đúng một lần/checkpoint theo protocol.
- [ ] Accuracy/Precision/Recall/F1/confusion matrix/FP/FN thật.
- [ ] Curves, Grad-CAM TP/TN/FP/FN và error analysis thật.
- [ ] Benchmark warm-up/mean/p50/p95 có môi trường thật.
- [ ] Ảnh thực tế báo cáo riêng, có cỡ mẫu.
- [ ] `report_facts.json` và evidence map hoàn chỉnh.

## Gate E - Submission-ready

- [ ] CLI/evaluate/UI khớp cùng ảnh/model/metadata.
- [ ] Demo chạy với checkpoint chính thức; lỗi file/model thân thiện.
- [ ] README/DATA_CARD/MODEL_CARD cập nhật artifact thật.
- [ ] Không placeholder URL/tên nhóm/TODO trong hồ sơ phát hành.
- [ ] Checkpoint và metadata có SHA-256; cách tải/cung cấp rõ.
- [ ] Secret scan và rà `.gitignore` pass.
- [ ] Clone/copy sạch, cài dependency, test và demo lại thành công.
- [ ] Bundle có source, config, manifest, model, results, docs; không nhét dataset nếu không cần.
- [ ] Báo cáo Word/PDF 35-45 trang hoàn chỉnh và render sạch.
- [ ] Slide PPTX/PDF hoàn chỉnh, metric khớp facts.
- [ ] Source code, demo, báo cáo và slide đủ theo phiếu đề tài.

## Biên bản bàn giao

```text
Git commit/tag:
Manifest SHA-256:
Checkpoint SHA-256:
Config SHA-256:
Selection lock:
Final run ID:
Report facts SHA-256:
Verification log:
Bundle SHA-256:
Ngày/người xác minh:
Trạng thái Accuracy >=0,92 (threshold nào):
Blocker còn lại:
```
