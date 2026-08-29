# Ma trận truy vết yêu cầu

Trạng thái: `CODE` = đã có implementation nhưng còn thiếu bằng chứng chính thức; `VERIFIED_CODE` = code đã qua test/smoke kỹ thuật; `VERIFIED_DATA` = yêu cầu dữ liệu cụ thể đã có bằng chứng trên archive chính thức; `BLOCKED_EXTERNAL` = cần dữ liệu/ảnh/tài liệu thật từ ngoài repository. `VERIFIED_CODE` hoặc `VERIFIED_DATA` không đồng nghĩa toàn bộ đồ án đã hợp lệ để nộp.

| ID | Yêu cầu | Module/entrypoint thực tế | Test/bằng chứng hiện có | Artifact chính thức còn phải có | Trạng thái |
|---|---|---|---|---|---|
| R01 | Crack=1, Non-crack=0, sigmoid=P(Crack) | `constants.py`, `model.py`, `service.py` | `test_data_manifest.py`, `test_model.py`, smoke delta CLI/service = 0 | metadata/checkpoint chính thức | VERIFIED_CODE |
| R02 | MobileNetV2 224x224, GAP, Dropout .3, sigmoid | `modeling/model.py` | `test_model.py`, smoke train/save/load | model summary E1-E4 thật | VERIFIED_CODE |
| R03 | preprocess dùng chung, RGB/EXIF/limits | `data/audit.py`, `data/pipeline.py`, `inference/preprocessing.py` | `test_data_pipeline.py`, `test_inference_preprocessing.py`, `test_inference_service.py` | preprocessing facts từ run thật | VERIFIED_CODE |
| R04 | SDNET download/checksum/safe extract | `scripts/download_sdnet2018.py` | `test_data_download.py`; native ZIP 528.286.896 byte, MD5 khớp `677411e...d0d7c6` | lưu provenance archive khi đóng gói nội bộ | VERIFIED_DATA |
| R05 | audit label/surface/file/hash/duplicates | `data/audit.py`, `data/manifest.py`, `scripts/audit_data.py`, `scripts/curate_manifest.py` | audit 56.092 hợp lệ/0 lỗi; curation loại 4 dòng/2 hash, clean 56.088, canonical `2ffb560a...880c7` | archive cùng audit/curation bundle khi bàn giao nội bộ | VERIFIED_DATA |
| R06 | source_group verified, fail-fast nếu thiếu | `data/manifest.py`, `data/split.py` | native manifest: 56.092 dòng verified, 230 nhóm D54/P104/W72; test guard thiếu group | snapshot quy tắc/hash cùng split chính thức | VERIFIED_DATA |
| R07 | split 70/15/15 seed42, không leakage | `data/split.py`, `scripts/create_splits.py` | locked `split_v1`: 39.014/8.540/8.534 ảnh; zero path/group/hash overlap; canonical `b38302d...99a3da7` | lưu bundle cùng hồ sơ nội bộ | VERIFIED_DATA |
| R08 | augmentation chỉ train, class weight từ train | `data/pipeline.py`, `scripts/visualize_augmentation.py` | `test_data_pipeline.py` | grid trước/sau E4 thật | VERIFIED_CODE |
| R09 | E1 frozen LR1e-3 không aug | config E1, `model.py`, `train.py` | config/model/training protocol tests | E1 run thật | VERIFIED_CODE |
| R10 | E2 block_14 LR1e-4 BN frozen | config E2, `model.py`, `train.py` | layer/trainability/training protocol tests | E2 run thật | VERIFIED_CODE |
| R11 | E3 block_10 LR1e-5 scheduler | config E3, `model.py`, `train.py` | layer/BN/callback/resume tests | E3 run thật | VERIFIED_CODE |
| R12 | E4 chọn E2/E3 chỉ validation + aug | `select_model.py`, `reporting/aggregate.py`, `train.py` | `test_reporting_aggregate.py`; no-test selection guard | `model_selection.json` + E4 thật | VERIFIED_CODE |
| R13 | E5 tune validation F1/tie-break | `modeling/threshold.py`, `lock_selection.py` | `test_threshold.py`, `test_selection.py` | threshold curve/lock E4 thật | VERIFIED_CODE |
| R14 | final evaluator khóa hash/test | `modeling/evaluate.py`, `evaluate_final.py` | `test_evaluate.py`; one-pass smoke final + immutable registry | final evaluation từng checkpoint thật | VERIFIED_CODE |
| R15 | Accuracy target >=.92, metric đầy đủ | `metrics.py`, `evaluate.py`, report aggregate | `test_metrics.py`, `test_reporting_aggregate.py` | `report_facts.json` và metrics test thật | CODE |
| R16 | Grad-CAM target Crack/out_relu | `gradcam.py`, `generate_gradcam_grid.py` | `test_gradcam_overlay.py`; smoke heatmap 7×7/overlay | grid TP/TN/FP/FN test thật | CODE |
| R17 | CLI/evaluate/UI nhất quán | `service.py`, `predict.py`, `app.py` | smoke CLI/service delta = 0; service tests | đối chiếu checkpoint chính thức | VERIFIED_CODE |
| R18 | Streamlit upload/result/warning | `app.py`, inference service | AppTest/health 200; service valid/invalid upload pass | demo checkpoint chính thức + upload thủ công | VERIFIED_CODE |
| R19 | latency <=5s được đo trung thực | `benchmark.py`, `benchmark_inference.py` | `test_reporting_benchmark.py` | benchmark checkpoint/môi trường thật | CODE |
| R20 | ảnh thực tế tách riêng | `real_images.py`, `evaluate_real_images.py` | guard schema/split được cài đặt | ảnh nhóm tự chụp + manifest + metrics riêng | BLOCKED_EXTERNAL |
| R21 | run bất biến, provenance/hash/environment | `train.py`, `selection.py`, `evaluate.py`, utils | overwrite/hash/one-shot + interruption→resume tests; smoke provenance | run E1-E5 thật có Git commit | VERIFIED_CODE |
| R22 | README/cards/notebook/handoff | root docs + standalone `CrackSpot.ipynb` | notebook 27 cell tự chứa code; default-path execution, real-image batch/model/fine-tune/Grad-CAM và Streamlit-source smoke pass | manual clean-copy/release review | VERIFIED_CODE |
| R23 | báo cáo 35-45 trang và slide | handoff riêng | đối chiếu report_facts | report/slide final | BLOCKED_EXTERNAL |

## Bằng chứng Gate B hiện tại

- `ruff check .`: pass ngày 2026-08-29.
- `pytest -q`: `260 passed` ngày 2026-08-29; vẫn phải chạy lại ở lần đóng gói cuối.
- `pip check`: `No broken requirements found`.
- Smoke: `artifacts/verification/smoke_pipeline.json`, run `smoke-20260829-verified`, exit 0, trạng thái `NOT_VALID_FOR_REPORT`.
- Streamlit: `artifacts/verification/streamlit_smoke.json`, health HTTP 200 và AppTest không exception; phiên này không có browser backend nên không tuyên bố browser upload pass.
- Checkpoint smoke SHA-256 `0149f30f6012960bad04fabb0df4511d664388aa276644e19b44c28447c9549d` chỉ là bằng chứng kỹ thuật, không phải model nộp.

## Bằng chứng Gate C hiện tại

- Native archive MD5: `677411e784f194422c90f52d9ed0d7c6`, đã khớp nguồn.
- `data/manifests/data_audit.json`: 56.092 file hợp lệ, 0 file lỗi; 8.484 Crack, 47.608 Non-crack.
- `data/manifests/audit_manifest.csv`: canonical SHA-256 `3b2acbaddb5431726c08bc4562d3f515210f0827fbbc057e1c08c9dd70369f5f`.
- Source groups: 230 nhóm đã xác minh bằng tiền tố trước dấu `-`; D=54, P=104, W=72.
- Curation chính thức `pre_split_curation_v1` đã khóa: loại 4 dòng thuộc 2 hash mâu thuẫn; giữ 56.088 dòng, 230 nhóm; 0 hash xung đột; clean canonical SHA-256 `2ffb560a49aadfe475129ca56b0ae90741c37913a4657dd60cd51839cea880c7`.
- `split_v1` đã khóa: train/validation/test = 39.014/8.540/8.534 ảnh và 160/35/35 nhóm; canonical SHA-256 `b38302d62547fb20f264b811f38f7031cf1b8d4c8a21f9db15d4dc39699a3da7`; zero path/group/hash overlap; đủ sáu strata mỗi split.
- Full byte-integrity scan pass 56.088 ảnh/530.749.950 byte, fingerprint `7bed031907031ba302d42fc1349b522d11300ff4f54a4740bae56a3d92caa4fd`. Bằng chứng: `artifacts/verification/gate_c_split_v1.json`.
- Phần SDNET2018 của Gate C đã đạt; ảnh tự chụp vẫn là `BLOCKED_EXTERNAL` theo R20.

## Cập nhật ma trận

Sau mỗi gate:

1. đổi trạng thái có dẫn đường dẫn cụ thể;
2. ghi tên test và kết quả, không chỉ “đã test”;
3. ghi run ID/hash cho bằng chứng thực nghiệm;
4. giữ `BLOCKED_EXTERNAL` cho ảnh tự chụp/GPU/báo cáo nếu chưa có;
5. không dùng smoke làm bằng chứng R15/R20/R23.
