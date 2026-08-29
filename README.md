# CrackSpot

**Xây dựng Hệ thống Phát hiện Vết nứt trên Bề mặt Tường/Đường bằng CNN**<br>
**CrackSpot: A CNN-based Surface Crack Detection System**

CrackSpot là đồ án phân loại ảnh bề mặt thành hai lớp bằng MobileNetV2 transfer learning:

- `Crack = 1` - ảnh có vết nứt;
- `Non-crack = 0` - ảnh không có vết nứt;
- đầu ra sigmoid luôn được hiểu là xác suất của lớp `Crack`.

Demo hiển thị nhãn, xác suất Crack, ngưỡng phân loại, thời gian suy luận và Grad-CAM chồng lên ảnh gốc. Grad-CAM chỉ là **bản đồ nhiệt vùng mô hình chú ý/vùng nghi ngờ**, không phải mask phân đoạn hay định vị chính xác theo pixel.

> **Cảnh báo an toàn:** CrackSpot chỉ hỗ trợ khảo sát sơ bộ. Kết quả không thay thế đánh giá của kỹ sư xây dựng, không đo kích thước vết nứt và không kết luận mức độ nguy hiểm hay phương án bảo trì.

## Trạng thái bằng chứng

**Trạng thái hiện tại: CHƯA SẴN SÀNG NỘP.** Repository tách rõ mã nguồn với kết quả thực nghiệm. Chỉ số chính thức chỉ được công bố khi có `artifacts/report/final_bundle_v1/report_facts.json`, kèm hash checkpoint và manifest đã khóa. Hiện chưa có metric E1-E5 chính thức; không được lấy mục tiêu `Accuracy >= 0,92`, output smoke hoặc metric validation làm kết quả.

Phần SDNET2018 của Gate C đã hoàn tất: MD5 native khớp nguồn, 56.092/56.092 ảnh decode hợp lệ, 230 `source_group` đã kiểm chứng; bốn dòng thuộc hai exact hash có nhãn mâu thuẫn đã được loại trước split. Locked bundle `split_v1` chứa 56.088 dòng, canonical SHA-256 `b38302d62547fb20f264b811f38f7031cf1b8d4c8a21f9db15d4dc39699a3da7`; train/validation/test lần lượt có 39.014/8.540/8.534 ảnh và 160/35/35 nhóm. Audit xác nhận đủ sáu strata ở mỗi split và không giao path/group/hash. Full byte-integrity scan 56.088 ảnh cũng pass.

Các phụ thuộc còn phải hoàn tất:

1. tạo Git commit sạch rồi huấn luyện E1-E4 trên cùng split cố định;
2. khóa E5 và final test đúng protocol;
3. ảnh thực tế tự chụp có manifest/nhãn;
4. báo cáo 35-45 trang và slide đối chiếu số liệu thật.

## Phạm vi

### Có trong dự án

- audit SDNET2018, phát hiện file lỗi/trùng và sinh manifest;
- split `70/15/15`, seed `42`, theo `source_group` để chống leakage;
- MobileNetV2 ImageNet, đầu vào RGB `224 x 224`;
- head `GlobalAveragePooling -> Dropout(0.3) -> Dense(1, sigmoid)`;
- E1-E4 huấn luyện/fine-tuning; E5 chọn threshold trên validation;
- Accuracy, Precision/Recall/F1 của lớp Crack, macro average, confusion matrix;
- benchmark latency và phân tích TP/TN/FP/FN bằng Grad-CAM;
- CLI và demo Streamlit dùng chung package `src/crackspot`.

### Ngoài phạm vi

- đăng nhập, phân quyền, thanh toán, cơ sở dữ liệu và lịch sử người dùng;
- object detection hoặc semantic segmentation khi không có box/mask thật;
- đo chiều rộng/độ sâu, đánh giá an toàn kết cấu hay đề xuất bảo trì;
- lưu lâu dài ảnh người dùng tải lên.

## Kiến trúc ngắn gọn

```text
SDNET2018 -> audit/manifest -> split khóa -> tf.data -> MobileNetV2
                                              |          |
                                              |          +-> .keras + metadata
                                              |                       |
                                              +-> evaluate/report <---+
                                                                      |
Ảnh JPG/PNG -> decode/EXIF/RGB -> preprocess chung -> inference -> Grad-CAM -> Streamlit/CLI
```

Chi tiết: [`docs/architecture.md`](docs/architecture.md), protocol: [`docs/experiment_protocol.md`](docs/experiment_protocol.md), traceability: [`docs/requirements_traceability.md`](docs/requirements_traceability.md).

## Cấu trúc repository

```text
CrackSpot/
├── CrackSpot.ipynb              # entrypoint/bảng điều khiển chính
├── app.py                       # entrypoint Streamlit
├── configs/                     # base và E1-E5
├── src/crackspot/               # logic dùng chung
├── scripts/                     # entrypoint dữ liệu/thí nghiệm/báo cáo
├── tests/                       # unit/integration/fixture nhỏ
├── data/                        # dataset/manifest cục bộ, không commit
├── models/                      # checkpoint chính thức hoặc hướng dẫn lấy
├── artifacts/                   # run/report/verification, phần lớn không commit
└── docs/                        # cài đặt, sử dụng, demo, bàn giao
```

## Chạy bằng notebook chính

Mở [`CrackSpot.ipynb`](CrackSpot.ipynb) bằng VS Code/Jupyter và chạy từ trên xuống. Chỉ cần sửa cell **THAM SỐ ĐIỀU KHIỂN**; các cờ `DO_*` mặc định đều tắt nên không tự train hay mở final test. Notebook là bảng điều khiển duy nhất, còn `src/crackspot` và `scripts/` là engine để tránh lặp logic.

Trên Colab, mở chính file `CrackSpot.ipynb`; URL repository và chế độ clone/cài dependency tự động đã được cấu hình sẵn. Chỉ cần bật `DO_DOWNLOAD_SDNET2018`, sau đó chạy từ trên xuống. Notebook clone source/locked manifests vào `/content/CrackSpot`; archive/ảnh SDNET2018 không đẩy lên Git mà được tải từ nguồn chính thức, xác minh MD5 rồi giải nén trong runtime. `INSTALL_DEPENDENCIES=None` nghĩa là tự cài trên Colab và không tự cài ở local.

Cell inventory hiển thị toàn bộ tệp source/config/docs/manifests, trạng thái Git, số lượng ảnh trong sáu thư mục nguồn và preview `train.csv`/`validation.csv`/`test.csv`. Notebook không in 56.088 tên ảnh vì output đó quá lớn và có thể làm treo trình duyệt.

## Môi trường mục tiêu

| Môi trường | Python | TensorFlow | Gia tốc |
|---|---:|---:|---|
| Windows 10/11 native | 3.12 | 2.19.0 | CPU |
| Google Colab | 3.13 | 2.21.0 | GPU do Colab cung cấp |
| WSL2 Ubuntu | 3.12 | 2.19.0 | GPU khi driver/runtime tương thích |

TensorFlow native Windows sau 2.10 không hỗ trợ GPU chính thức; dùng WSL2 hoặc Colab nếu cần GPU. Mỗi run phải ghi lại môi trường thực tế trong `environment.json`; bảng trên là target, không phải tuyên bố đã benchmark.

Bộ pin tự chọn theo Python: TensorFlow 2.19.0/h5py 3.16.0 cho Python 3.10-3.12 và TensorFlow 2.21.0/h5py 3.14.0 cho Python 3.13 của Colab. Đây là sai khác có chủ ý so với ưu tiên TensorFlow 2.15.x/Python 3.10-3.11 trong đề cương; mỗi run vẫn ghi lại chính xác phiên bản thực tế trong artifact để bảo đảm khả năng tái lập.

Khuyến nghị tối thiểu cho full experiment: 16 GB RAM, khoảng 15 GB dung lượng trống cho archive/dataset/artifact, và GPU có VRAM phù hợp nếu muốn giảm thời gian. CPU vẫn dùng được cho test, smoke và demo nhưng full training có thể lâu.

## Cài đặt

### Windows PowerShell - CPU/demo

```powershell
cd D:\Đồ án\CrackSpot
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

### Windows PowerShell - phát triển và huấn luyện CPU

```powershell
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

### Google Colab - huấn luyện GPU

Mở `CrackSpot.ipynb` và chạy tuần tự từ đầu. Notebook sẽ tự clone repository, nâng cấp công cụ cài đặt và chọn đúng TensorFlow theo phiên bản Python; không tạo virtual environment trong Colab.

### WSL2 - huấn luyện GPU

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-train.txt
python -m pip install -e .
```

Kiểm tra TensorFlow thấy thiết bị nào:

```powershell
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices())"
```

Xem hướng dẫn đầy đủ và lỗi thường gặp tại [`docs/installation.md`](docs/installation.md).

## Dữ liệu SDNET2018

- DOI dataset: <https://doi.org/10.15142/T3TD19>
- Trang nguồn: <https://digitalcommons.usu.edu/all_datasets/48/>
- Bài báo mô tả: <https://doi.org/10.1016/j.dib.2018.11.015>
- Giấy phép dữ liệu: CC BY 4.0.
- MD5 gói native đã tải và đã xác minh khớp nguồn: `677411e784f194422c90f52d9ed0d7c6` (528.286.896 byte).
- Audit ngày 2026-08-29: 56.092 ảnh JPG RGB 256 x 256 hợp lệ, 0 file lỗi; 8.484 Crack và 47.608 Non-crack.
- Quy tắc nhóm là tiền tố tên file trước dấu `-`; toàn bộ 56.092 dòng đã được xác minh, gồm 230 `source_group`: D=54, P=104, W=72.
- Canonical SHA-256 của manifest audit gốc: `3b2acbaddb5431726c08bc4562d3f515210f0827fbbc057e1c08c9dd70369f5f`.

Trang Digital Commons hiện có nhiều attachment. Downloader mặc định lấy **native bundle** `DATA_Maguire_20180517_ALL.zip` (lưu cục bộ là `data/raw/SDNET2018.zip`) vì đây là file khớp MD5 công bố; bundle chứa README và một ZIP dataset lồng bên trong. Không thay URL bằng attachment “additional” chỉ vì đó cũng là ZIP hợp lệ: file đó không mang MD5 công bố của native bundle.

Không commit dataset. Tải/audit bằng script:

```powershell
python scripts/download_sdnet2018.py --help
python scripts/download_sdnet2018.py `
  --archive data/raw/SDNET2018.zip `
  --extract-dir data/raw
python scripts/audit_data.py data/raw/SDNET2018 `
  --manifest-out data/manifests/audit_manifest.csv `
  --report-out data/manifests/data_audit.json
```

Audit gốc phát hiện 13 nhóm exact-duplicate gồm 26 ảnh. Trong đó hai hash có nhãn mâu thuẫn và đã chặn bước split ban đầu:

- `D/CD/7039-112.jpg` (`Crack`) và `D/UD/7039-112_2.jpg` (`Non-crack`);
- `W/CW/7074-105.jpg` (`Crack`) và `W/UW/7074-105_2.jpg` (`Non-crack`).

Đây là lỗi chất lượng dữ liệu **trước split**, không phải loại mẫu sau khi xem test. Chính sách bắt buộc là giữ nguyên `audit_manifest.csv`; loại toàn bộ bốn dòng thuộc hai hash mâu thuẫn khỏi manifest tiền split; sinh conflict report ghi hash, path, nhãn gốc, lý do, canonical hash của manifest cha và hash manifest đã làm sạch; công bố số dòng bị loại và phân bố sau làm sạch. Không tự sửa nhãn, không chỉ giữ một bản tùy ý và không xóa ảnh khỏi archive gốc.

Bộ curation chính thức đã được tạo bằng lệnh sau; output cùng tên không được ghi đè hoặc tái sinh:

```powershell
python scripts/curate_manifest.py data/manifests/audit_manifest.csv `
  --output-dir data/manifests/pre_split_curation_v1
```

Output đã khóa gồm:

- `conflict_rows.csv`: 4 dòng bị loại, hai Crack và hai Non-crack;
- `conflict_report.json`: policy, parent/artifact hashes, hai nhóm xung đột và validation;
- `pre_split_manifest.csv`: 56.088 dòng, 8.482 Crack, 47.606 Non-crack, 230 nhóm đã xác minh và 0 exact hash mâu thuẫn.

Canonical SHA-256 của manifest làm sạch là `2ffb560a49aadfe475129ca56b0ae90741c37913a4657dd60cd51839cea880c7`; SHA-256 bytes của file là `a5e37042660f7d42ac8e526545e111beb020d53e76d9ecffbb8d8d2228ac4b2b`. Mười một nhóm exact-duplicate cùng nhãn (22 dòng) được giữ nguyên và khóa cùng một split theo exact hash. Official bundle đã được tạo bằng lệnh:

```powershell
python scripts/create_splits.py data/manifests/pre_split_curation_v1/pre_split_manifest.csv `
  --output-dir data/manifests/split_v1 `
  --conflict-report data/manifests/pre_split_curation_v1/conflict_report.json `
  --seed 42
```

`verify_locked_split_bundle()` xác minh completion marker, inventory, snapshot lineage, seed/tỷ lệ, hash và audit. Canonical SHA-256 của `split_v1/manifest.csv` là `b38302d62547fb20f264b811f38f7031cf1b8d4c8a21f9db15d4dc39699a3da7`; full byte scan kiểm tra 530.749.950 byte với fingerprint `7bed031907031ba302d42fc1349b522d11300ff4f54a4740bae56a3d92caa4fd`. Bằng chứng tóm tắt nằm tại `artifacts/verification/gate_c_split_v1.json`.

`configs/base.yaml` vẫn để `group_rule_verified: false` làm mặc định an toàn cho dữ liệu chưa kiểm chứng; manifest audit chính thức ghi `source_group_verified=True` cho toàn bộ dòng. Full split phải tiếp tục fail-fast nếu thiếu cờ xác minh hoặc còn exact hash mâu thuẫn.

Kết quả bắt buộc sau curation/split:

- manifest có `relative_path,label,surface,source_group,sha256,width,height,split`;
- split train/validation/test bất biến;
- `split_audit.json` chứng minh không giao path/source_group/exact hash;
- hash SHA-256 của manifest làm sạch và từng manifest split;
- liên kết provenance ngược về canonical SHA-256 của manifest audit gốc.

Ảnh tự chụp đặt tại `data/external/real/` và có manifest/nhãn riêng. Không gộp vào test chuẩn. Xem [`DATA_CARD.md`](DATA_CARD.md).

## Chạy E1-E5

Mọi run có `run_id` riêng và không ghi đè artifact. Xem tùy chọn chính xác bằng `--help` vì entrypoint là nguồn sự thật của CLI.

```powershell
python scripts/run_experiment.py --config configs/experiments/e1_baseline.yaml --run-id E1_RUN_ID
python scripts/run_experiment.py --config configs/experiments/e2_finetune_basic.yaml --run-id E2_RUN_ID
python scripts/run_experiment.py --config configs/experiments/e3_finetune_deep.yaml --run-id E3_RUN_ID

python scripts/select_model.py `
  --e2-run artifacts/runs/E2_RUN_ID `
  --e3-run artifacts/runs/E3_RUN_ID `
  --output artifacts/model_selection.json

python scripts/run_experiment.py `
  --config configs/experiments/e4_augmentation.yaml `
  --model-selection artifacts/model_selection.json `
  --run-id E4_RUN_ID

python scripts/visualize_augmentation.py `
  --config artifacts/runs/E4_RUN_ID/config_snapshot.json `
  --manifest data/manifests/split_v1/manifest.csv `
  --dataset-root data/raw/SDNET2018 `
  --output artifacts/runs/E4_RUN_ID/augmentation_before_after.png
```

Ý nghĩa:

- **E1:** backbone frozen, LR `1e-3`, không augmentation;
- **E2:** head rồi unfreeze từ `block_14_expand`, BN frozen, LR fine-tune `1e-4`;
- **E3:** head rồi unfreeze từ `block_10_expand`, BN frozen, LR fine-tune `1e-5`, theo dõi overfitting;
- **E4:** chọn cấu hình E2/E3 chỉ theo validation rồi chạy lại với augmentation đã khóa;
- **E5:** không phải model mới; tối ưu threshold của checkpoint E4 trên validation bằng F1 Crack, tie-break Recall rồi khoảng cách tới 0.5.

Khi runtime bị ngắt, chỉ resume đúng `run_id`, config và manifest ban đầu; không dùng resume để thay đổi split, threshold hay siêu tham số giữa run:

```powershell
python scripts/run_experiment.py `
  --config configs/experiments/e2_finetune_basic.yaml `
  --manifest data/manifests/split_v1/manifest.csv `
  --dataset-root data/raw/SDNET2018 `
  --run-id E2_RUN_ID `
  --resume
```

Resume từ chối run đã hoàn tất, config/manifest/hash lệch hoặc evidence dở dang xung đột; phase đã có completion marker hợp lệ sẽ không bị train lại. Xem chữ ký hiện hành bằng `python scripts/run_experiment.py --help`.

### Khóa lựa chọn và final test

Test chỉ được mở sau khi có `selection_complete.json` chứa checkpoint/config/manifest hash và threshold đã khóa:

```powershell
$E4_RUN = "artifacts/runs/E4_RUN_ID"
$MANIFEST = "data/manifests/split_v1/manifest.csv"
$DATASET = "data/raw/SDNET2018"

crackspot-threshold `
  --predictions "$E4_RUN/predictions_validation.csv" `
  --output "$E4_RUN/threshold_validation.json"

python scripts/lock_selection.py `
  --run-dir $E4_RUN `
  --experiment E5 `
  --threshold-result "$E4_RUN/threshold_validation.json"

python scripts/export_selected_metadata.py `
  --selection "$E4_RUN/selection_complete.json" `
  --output "$E4_RUN/selected_model.metadata.json"

python scripts/evaluate_final.py `
  --selection "$E4_RUN/selection_complete.json" `
  --manifest $MANIFEST `
  --dataset-root $DATASET `
  --output-dir artifacts/report/final_evaluation/E4_RUN_ID

python scripts/benchmark_inference.py data/external/real/probe.jpg `
  --model "$E4_RUN/model.keras" `
  --metadata "$E4_RUN/selected_model.metadata.json" `
  --output artifacts/benchmarks/E4_RUN_ID.json

python scripts/generate_gradcam_grid.py `
  --selection "$E4_RUN/selection_complete.json" `
  --predictions artifacts/report/final_evaluation/E4_RUN_ID/predictions_test.csv `
  --dataset-root $DATASET `
  --output artifacts/report/fig_gradcam_tp_tn_fp_fn.png
```

Thay các tên `*_RUN_ID` và ảnh `probe.jpg` bằng artifact/ảnh thật của lần chạy. Với E1-E3, khóa threshold `0.5` bằng `lock_selection.py --threshold 0.5`, rồi đánh giá đúng một lần cho từng checkpoint. Không đưa output smoke vào `artifacts/report/`.

Đánh giá ảnh do nhóm tự chụp bằng lệnh riêng; lệnh này yêu cầu manifest khai báo `capture_source=self_captured` và tuyệt đối không nhập các mẫu đó vào test SDNET2018:

```powershell
python scripts/evaluate_real_images.py `
  --selection "$E4_RUN/selection_complete.json" `
  --manifest data/external/real/manifest.csv `
  --dataset-root data/external/real `
  --output-dir artifacts/report/real_images/E4_RUN_ID `
  --confirm-self-captured
```

Sau khi cả final evaluation chuẩn và đánh giá ảnh tự chụp đã hoàn tất, tổng hợp report assets bằng đúng hai nguồn tách biệt:

```powershell
python scripts/generate_report_assets.py `
  --evaluation-dir artifacts/report/final_evaluation/E4_RUN_ID `
  --manifest $MANIFEST `
  --validation-predictions "$E4_RUN/predictions_validation.csv" `
  --benchmark artifacts/benchmarks/E4_RUN_ID.json `
  --real-evaluation-dir artifacts/report/real_images/E4_RUN_ID `
  --output-dir artifacts/report/final_bundle_v1
```

E1-E4 báo cáo test ở threshold `0.5` theo protocol; checkpoint E4 tốt nhất được báo cáo thêm ở threshold E5. Không tune trên test và không đánh giá lặp checkpoint để chọn kết quả đẹp.

## Dự đoán và demo

Đặt hai tệp checkpoint chính thức:

```text
models/crackspot.keras
models/crackspot.metadata.json
```

Hoặc đặt biến môi trường:

```powershell
$env:CRACKSPOT_MODEL_PATH = "D:\duong-dan\crackspot.keras"
$env:CRACKSPOT_METADATA_PATH = "D:\duong-dan\crackspot.metadata.json"
```

Xác minh SHA-256 theo giá trị đã công bố trong metadata/release:

```powershell
Get-FileHash .\models\crackspot.keras -Algorithm SHA256
```

CLI:

```powershell
python scripts/predict.py .\duong-dan\anh.jpg
```

Streamlit:

```powershell
streamlit run app.py
```

Ứng dụng nhận JPG/JPEG/PNG, mặc định tối đa 10 MB và 25 megapixel, decode bytes thật, tôn trọng EXIF orientation, chuyển grayscale/RGBA sang RGB và xử lý in-memory. Xem [`docs/user_guide.md`](docs/user_guide.md) và kịch bản bảo vệ tại [`docs/demo_script.md`](docs/demo_script.md).

## Kiểm thử và smoke

Các kiểm thử không được phụ thuộc dataset/checkpoint/network thật:

```powershell
ruff check .
ruff format --check .
pytest -q
```

Chạy smoke end-to-end theo CLI hiện có và lưu kết quả dưới `artifacts/smoke/` với cờ `NOT_VALID_FOR_REPORT`. Dữ liệu synthetic/tiny tuyệt đối không được trộn vào `artifacts/report/`.

Sau khi có checkpoint chính thức, đối chiếu cùng một ảnh giữa CLI, evaluation và UI; chạy Streamlit health/smoke; lưu lệnh, thời gian và kết quả trong `artifacts/verification/`.

## Artifact và kết quả

Mỗi run cần lưu config snapshot/hash, manifest hash, môi trường, seed/deterministic, log, `history.csv`, checkpoint/hash, metrics/predictions validation, tham số và thời gian. `metrics_test.json` và `predictions_test.csv` chỉ được sinh ở final evaluation.

`artifacts/report/final_bundle_v1/report_facts.json` là nguồn số liệu duy nhất cho báo cáo/slide. Không viết metric vào README, MODEL_CARD hay tài liệu nộp nếu chưa có artifact thật. Quy tắc bàn giao nằm tại [`artifacts/README.md`](artifacts/README.md) và [`docs/report_evidence_map.md`](docs/report_evidence_map.md).

## Quyền riêng tư, giới hạn và đạo đức

- Ảnh upload chỉ xử lý trong bộ nhớ theo luồng demo; không thiết kế lưu lịch sử.
- Không upload ảnh chứa dữ liệu nhạy cảm khi chạy trên dịch vụ bên thứ ba nếu chưa có quyền.
- SDNET2018 không đại diện mọi vật liệu, camera, ánh sáng hay môi trường địa phương.
- Mất cân bằng lớp có thể làm Accuracy gây hiểu nhầm; luôn đọc Recall/F1 Crack và FP/FN.
- Grad-CAM có độ phân giải thô và có thể chú ý vào nền/bóng/mối nối.
- Dự đoán ngoài miền dữ liệu có thể kém dù test chuẩn tốt.

## Bàn giao

- Báo cáo: [`docs/report_handoff.md`](docs/report_handoff.md)
- Slide: [`docs/slides_handoff.md`](docs/slides_handoff.md)
- Checklist nộp: [`docs/submission_checklist.md`](docs/submission_checklist.md)
- Theo dõi tuần: [`docs/weekly_progress_template.md`](docs/weekly_progress_template.md)

## Trích dẫn và giấy phép

Trích dẫn project theo [`CITATION.cff`](CITATION.cff). Mã nguồn được phát hành theo giấy phép MIT trong [`LICENSE`](LICENSE). Dataset SDNET2018 là tài sản riêng của tác giả/nguồn dataset và dùng theo CC BY 4.0; giấy phép mã nguồn không thay thế giấy phép dữ liệu hay trọng số bên thứ ba.
