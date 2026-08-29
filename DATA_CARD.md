# Data Card - SDNET2018 trong CrackSpot

## Tóm tắt

CrackSpot dùng **SDNET2018** làm nguồn dữ liệu công khai chính cho bài toán phân loại nhị phân ảnh bề mặt:

- `Crack = 1`;
- `Non-crack = 0`.

Ảnh thực tế tự chụp là một tập ngoài miền dữ liệu, được đánh giá riêng sau khi khóa mô hình. Không gộp ảnh tự chụp vào train/validation/test chuẩn.

**Trạng thái:** audit archive, curation và locked split SDNET2018 đã xác minh. Phần dữ liệu SDNET công khai của Gate C đã đạt; ảnh thực tế tự chụp vẫn chưa có và toàn dự án vẫn **CHƯA SẴN SÀNG NỘP**.

## Nguồn, trích dẫn và giấy phép

- Dataset DOI: <https://doi.org/10.15142/T3TD19>
- Trang phát hành: <https://digitalcommons.usu.edu/all_datasets/48/>
- Bài báo: Dorafshan, Thomas và Maguire, *Data in Brief* (2018), DOI <https://doi.org/10.1016/j.dib.2018.11.015>
- Giấy phép dữ liệu được trang nguồn công bố: **CC BY 4.0**.
- MD5 gói ZIP native được nguồn công bố và đã xác minh trên file tải về: `677411e784f194422c90f52d9ed0d7c6` (528.286.896 byte).

Giấy phép MIT của source CrackSpot không áp dụng cho dataset. Khi phân phối lại dữ liệu hoặc hình lấy từ dữ liệu, phải giữ attribution và tuân thủ CC BY 4.0.

## Audit archive chính thức

Audit ngày 2026-08-29 trên nội dung giải nén từ native bundle cho kết quả:

| Thuộc tính | Kết quả audit |
|---|---:|
| File phát hiện / hợp lệ / lỗi | 56.092 / 56.092 / 0 |
| Ảnh RGB JPEG 256 x 256 | 56.092 |
| Crack (`1`) | 8.484 |
| Non-crack (`0`) | 47.608 |
| Bề mặt D / P / W | 13.620 / 24.334 / 18.138 |
| `source_group` D / P / W | 54 / 104 / 72 |
| Tổng `source_group` đã xác minh | 230 |

Các số trên khớp tổng số ảnh/nhãn công bố. Quy tắc `source_group` đã kiểm chứng là tiền tố tên patch trước dấu `-`; không có collision tiền tố giữa ba bề mặt. Toàn bộ 56.092 dòng có `source_group_verified=True`.

Bằng chứng gốc là `data/manifests/data_audit.json` và `data/manifests/audit_manifest.csv`. Canonical SHA-256 của manifest audit là `3b2acbaddb5431726c08bc4562d3f515210f0827fbbc057e1c08c9dd70369f5f`. Các artifact tổng hợp cho báo cáo sau này phải dẫn ngược về hash này; không thay số audit bằng số tham chiếu viết tay.

Native bundle chứa README và một ZIP dataset lồng bên trong. Checksum của ZIP lồng khác checksum native bundle là cấu trúc phát hành dự kiến, không phải lý do thay file native bằng attachment khác.

## Cấu trúc và ánh xạ nhãn

```text
SDNET2018/
├── D/
│   ├── CD/  # bridge deck, Crack=1
│   └── UD/  # bridge deck, Non-crack=0
├── P/
│   ├── CP/  # pavement, Crack=1
│   └── UP/  # pavement, Non-crack=0
└── W/
    ├── CW/  # wall, Crack=1
    └── UW/  # wall, Non-crack=0
```

Không suy nhãn từ từ khóa tùy ý ngoài sáu thư mục đã định nghĩa. File không khớp mapping phải được báo lỗi/audit.

## Audit bắt buộc

Trước khi split, pipeline phải:

1. xác minh checksum archive nếu dùng đúng gói nguồn;
2. giải nén an toàn, không cho path traversal;
3. decode từng ảnh thay vì chỉ tin extension;
4. ghi width, height, mode/kênh và file lỗi;
5. thống kê theo label và surface;
6. tính SHA-256 để phát hiện exact duplicate;
7. audit near-duplicate bằng perceptual hash nếu implementation/môi trường hỗ trợ;
8. ghi nguồn và phiên bản/thời điểm lấy dữ liệu.

Manifest tối thiểu:

```text
relative_path,label,surface,source_group,sha256,width,height,split
```

CSV/Parquet và mọi báo cáo phải dùng path tương đối, không ghi đường dẫn máy cá nhân.

## Mâu thuẫn nhãn exact-duplicate và curation tiền split

Audit phát hiện 13 nhóm exact-duplicate gồm 26 ảnh. Hai nhóm (bốn file) có cùng bytes nhưng nằm trong thư mục nhãn đối nghịch:

| SHA-256 | Crack | Non-crack |
|---|---|---|
| `b3ace75a69b5c5a28695dd9dcaddfa3b2efbf88ad533773dc1e0169a9f85520b` | `D/CD/7039-112.jpg` | `D/UD/7039-112_2.jpg` |
| `b85d1a85d3f5609e4d76cdfc2e21c064295ca0e0b688829764f836bd63a99bca` | `W/CW/7074-105.jpg` | `W/UW/7074-105_2.jpg` |

Không thể dùng các file này làm mẫu có nhãn đáng tin cậy, và splitter đúng thiết kế phải từ chối manifest còn exact hash mâu thuẫn. Chính sách curation trước split là:

1. giữ nguyên archive, `audit_manifest.csv` và `data_audit.json` làm bằng chứng nguồn;
2. loại **toàn bộ** dòng thuộc hai hash mâu thuẫn khỏi manifest tiền split, không tự sửa nhãn và không chọn giữ một phía;
3. sinh `conflict_rows.csv` và `conflict_report.json` ghi hash, mọi path/nhãn gốc, lý do loại, canonical hash manifest cha, thời điểm/quy tắc curation;
4. sinh `pre_split_manifest.csv` riêng, ghi số dòng/phân bố trước-sau và SHA-256 của chính nó;
5. chỉ tạo split từ manifest đã làm sạch, rồi audit lại path, `source_group` và exact hash.

Đây là xử lý chất lượng dữ liệu trước khi có train/validation/test, không phải cherry-picking test. Bộ `data/manifests/pre_split_curation_v1/` đã được tạo bất biến và xác minh:

| Thuộc tính sau curation | Giá trị |
|---|---:|
| Dòng giữ lại / loại | 56.088 / 4 |
| Crack / Non-crack giữ lại | 8.482 / 47.606 |
| Exact hash mâu thuẫn còn lại | 0 |
| `source_group` đã xác minh | 230 |
| Exact-duplicate cùng nhãn giữ lại | 11 nhóm / 22 dòng |

Canonical SHA-256 của `pre_split_manifest.csv` là `2ffb560a49aadfe475129ca56b0ae90741c37913a4657dd60cd51839cea880c7`, SHA-256 bytes của file là `a5e37042660f7d42ac8e526545e111beb020d53e76d9ecffbb8d8d2228ac4b2b`; canonical hash cha là `3b2acbaddb5431726c08bc4562d3f515210f0827fbbc057e1c08c9dd70369f5f`. Validation xác nhận chỉ các hash mâu thuẫn bị loại, không viết lại nhãn, mọi dòng cha được đối soát và chưa có split assignment trong lúc curation.

## Locked split chính thức

Bundle `data/manifests/split_v1/` dùng seed 42 và tỷ lệ mục tiêu 70/15/15 theo `source_group`:

| Split | Ảnh | Tỷ lệ thực tế | Source groups |
|---|---:|---:|---:|
| Train | 39.014 | 69,559% | 160 |
| Validation | 8.540 | 15,226% | 35 |
| Test | 8.534 | 15,215% | 35 |

Canonical SHA-256 của manifest đã split là `b38302d62547fb20f264b811f38f7031cf1b8d4c8a21f9db15d4dc39699a3da7`. `split_audit.json` xác nhận không giao path, `source_group` hoặc exact SHA-256 giữa mọi cặp split; mỗi split có đủ D/P/W × label 0/1 và nằm trong tolerance chính thức. Full byte-integrity scan kiểm tra 56.088 ảnh/530.749.950 byte, fingerprint `7bed031907031ba302d42fc1349b522d11300ff4f54a4740bae56a3d92caa4fd`.

## Source group và chống leakage

SDNET2018 gồm nhiều patch lấy từ 230 ảnh gốc. Patch từ cùng ảnh gốc có thể rất giống nhau; vì vậy random split theo patch có thể làm test quá lạc quan.

Quy tắc bắt buộc:

- với bản audit chính thức hiện tại, dùng tiền tố trước dấu `-` đã kiểm chứng; với nguồn khác, phải kiểm chứng lại hoặc dùng `group_map.csv` có provenance;
- toàn bộ patch cùng `source_group` phải ở đúng một split;
- không được coi clustering perceptual hash là thay thế tương đương cho source group;
- khi `data.group_rule_verified: false`, full experiment phải fail-fast và sinh báo cáo/template hỗ trợ xác minh;
- random split không group chỉ được dùng cho smoke, gắn `NOT_VALID_FOR_REPORT`.

## Split

Split chính thức đã được khóa tại `data/manifests/split_v1/` từ `pre_split_curation_v1/pre_split_manifest.csv`. Bundle đáp ứng:

- train: 39.014 ảnh (69,559%);
- validation: 8.540 ảnh (15,226%);
- test: 8.534 ảnh (15,215%);
- seed: 42;
- cân nhắc đồng thời label và surface trong giới hạn group split;
- lưu manifest bất biến và SHA-256 của từng manifest;
- `split_audit.json` chứng minh không giao nhau về path, `source_group` và exact hash.

Test không dùng để chọn cấu hình, checkpoint hay threshold. Không tái sinh split sau khi xem test.

## Tiền xử lý và augmentation

Pipeline dùng chung cho train/evaluate/CLI/demo:

- decode bytes thật, chặn decompression bomb, giới hạn mặc định 25 megapixel;
- tôn trọng EXIF orientation;
- grayscale/RGBA hợp lệ được chuyển về RGB;
- resize `224 x 224`;
- dùng `tf.keras.applications.mobilenet_v2.preprocess_input` để đưa `[0,255]` về `[-1,1]`.

Chỉ train được augmentation: horizontal flip `p=0.5`, rotation tối đa 15 độ, brightness và contrast nhẹ ở mức 0,15 theo config. Validation, test và ảnh thực tế không augmentation.

Balanced class weight được tính **chỉ từ train**, lưu trong artifact và áp dụng nhất quán E1-E4.

## Ảnh thực tế tự chụp

Đặt ảnh tại `data/external/real/` và tạo manifest riêng. Manifest nên có:

```text
relative_path,label,capture_date,location_category,surface,device,lighting,annotator,notes
```

Không đưa thông tin định vị hoặc người trong ảnh nếu không cần thiết. Nhãn cần được kiểm tra thủ công; cỡ mẫu phải công bố. Không tự sinh ảnh giả rồi gọi là ảnh thực tế. Kết quả của tập này chỉ phản ánh độ bền bước đầu trước domain shift, không thay test chuẩn.

## Thiên lệch và giới hạn

- Mất cân bằng lớp lớn có thể khiến Accuracy che giấu việc bỏ sót Crack.
- Các bề mặt, điều kiện ánh sáng, camera và kiểu vết nứt trong dataset không bao phủ mọi công trình.
- Patch 256 x 256 thiếu ngữ cảnh rộng của kết cấu.
- Nhãn chỉ ở mức toàn ảnh, không có box/mask để kiểm chứng định vị pixel.
- Bóng, mối nối, mép vá, vết bẩn, độ nhám và vật cản có thể gây false positive/negative.
- Kết quả trên dữ liệu Mỹ không mặc nhiên tổng quát cho điều kiện công trình Việt Nam.

Vì vậy, báo cáo phải trình bày Recall/F1 của Crack, FP/FN, kết quả theo surface khi đủ mẫu và đánh giá ảnh thực tế riêng.

## Provenance còn phải điền từ artifact thật

Audit gốc và curation đã được xác minh như trên. Không điền tay các giá trị split/thực nghiệm còn lại; lấy từ artifact tương ứng:

| Trường | Nguồn sự thật |
|---|---|
| Số file hợp lệ/lỗi và phân bố audit gốc | `data/manifests/data_audit.json` |
| Exact/near duplicates gốc | `data/manifests/data_audit.json` |
| Dòng loại và lý do | `pre_split_curation_v1/conflict_rows.csv` + `conflict_report.json` (đã khóa) |
| Phân bố manifest làm sạch | `pre_split_curation_v1/pre_split_manifest.csv` + `conflict_report.json` |
| Phân bố split | `dataset_summary.csv/json` sau split (chưa có) |
| Quy tắc source group | audit report + config snapshot |
| Hash manifest làm sạch/split | run metadata / `selection_complete.json` |
| Cỡ mẫu ảnh thực tế | real-image manifest/report |

Nếu thiếu bất kỳ bằng chứng cốt lõi nào ở một giai đoạn, Data Card phải ghi rõ giai đoạn đó **chưa xác minh**, không thay bằng thống kê dự kiến. Hiện audit gốc và curation đã xác minh; split vẫn chưa tạo/khóa.
