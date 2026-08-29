# Dữ liệu CrackSpot

Thư mục này lưu hướng dẫn và manifest đã review. Các manifest audit,
curation và locked split được commit để GitHub/Colab dùng đúng cùng một
split. Không commit ZIP, ảnh SDNET2018, ảnh thực tế riêng tư hoặc cache vào Git.

## Cấu trúc chuẩn đang dùng

```text
data/
├── raw/
│   ├── SDNET2018.zip          # archive nguồn đã khớp MD5
│   └── SDNET2018/             # dữ liệu gốc, giữ nguyên tên thư mục của nguồn
│       ├── D/{CD,UD}          # Deck: Crack / Non-crack
│       ├── P/{CP,UP}          # Pavement: Crack / Non-crack
│       └── W/{CW,UW}          # Wall: Crack / Non-crack
├── manifests/
│   ├── audit_manifest.csv
│   ├── data_audit.json
│   ├── pre_split_curation_v1/ # dữ liệu làm sạch trước chia tập
│   └── split_v1/              # split chính thức, bất biến
│       ├── train.csv
│       ├── validation.csv
│       ├── test.csv
│       ├── manifest.csv
│       ├── split_audit.json
│       ├── manifest_hashes.json
│       └── split_complete.json
└── external/real/             # ảnh nhóm tự chụp, đánh giá riêng
```

Train/validation/test được quản lý bằng ba manifest rõ tên, thay vì sao chép
56.088 ảnh sang ba cây thư mục khác nhau. Đây là cách tổ chức chuẩn cho split
theo `source_group`: giữ nguyên dữ liệu nguồn, tránh nhân đôi file và tránh vô
tình làm rò các patch cùng ảnh gốc. Loader lấy các dòng có `split=train`,
`split=validation` hoặc `split=test` từ locked manifest.

## Nguồn dữ liệu chính

- SDNET2018, DOI: <https://doi.org/10.15142/T3TD19>
- Trang phát hành: <https://digitalcommons.usu.edu/all_datasets/48/>
- Giấy phép: CC BY 4.0
- MD5 do trang phát hành công bố cho native bundle `DATA_Maguire_20180517_ALL.zip`
  (downloader lưu cục bộ là `SDNET2018.zip`):
  `677411e784f194422c90f52d9ed0d7c6`
- Thống kê tham chiếu: 56.092 ảnh con RGB 256×256 từ 230 ảnh gốc;
  8.484 `Crack` và 47.608 `Non-crack`. Audit thực tế mới là nguồn sự thật
  cho bản tải về đang dùng.

Quy ước nhãn duy nhất:

| Thư mục | Bề mặt | Nhãn | Tên lớp |
|---|---|---:|---|
| `D/CD` | bridge deck (`D`) | 1 | Crack |
| `D/UD` | bridge deck (`D`) | 0 | Non-crack |
| `P/CP` | pavement (`P`) | 1 | Crack |
| `P/UP` | pavement (`P`) | 0 | Non-crack |
| `W/CW` | wall (`W`) | 1 | Crack |
| `W/UW` | wall (`W`) | 0 | Non-crack |

## Quy trình tái tạo dữ liệu

Chạy từ thư mục gốc repository sau khi cài package editable:

```powershell
python scripts/download_sdnet2018.py
python scripts/audit_data.py data/raw/SDNET2018
```

Downloader truyền dữ liệu theo luồng, tiếp tục file `.part` khi máy chủ hỗ trợ
HTTP Range, xác minh MD5 và giải nén qua thư mục tạm. Script từ chối đường dẫn
ZIP traversal, symbolic link, archive vượt giới hạn và file đích đã tồn tại.
Nếu Digital Commons thay URL tải, mở trang phát hành chính thức và truyền URL mới
bằng `--url`; không tắt xác minh checksum.

Native bundle có README và một `SDNET2018.zip` lồng bên trong. Checksum công bố
áp dụng cho **bundle ngoài**, không phải attachment additional/ZIP trong. Script
xác minh bundle trước, kiểm tra an toàn cả hai tầng ZIP rồi xuất một root ổn
định `data/raw/SDNET2018/{D,P,W}`.

Lần audit đầu thường trả mã thoát `4`: ảnh đã được kiểm tra nhưng
`source_group` chưa được chứng minh. Đây là hành vi chủ đích. Audit tạo:

- `data/manifests/audit_manifest.csv`;
- `data/manifests/data_audit.json`;
- `data/manifests/group_map_template.csv` nếu chưa có ánh xạ nhóm.

Manifest ghi đường dẫn tương đối, nhãn, bề mặt, SHA-256 byte gốc, dHash để rà
soát near-duplicate, kích thước trước/sau EXIF, mode, định dạng, dung lượng, lỗi
audit và trạng thái xác minh nhóm. Ảnh hỏng không bị bỏ âm thầm mà có dòng
`audit_status=invalid`.

## Xác minh `source_group`

SDNET2018 gồm các patch cắt từ 230 ảnh gốc. Tất cả patch của một ảnh gốc phải
ở cùng split. Không được đoán một biểu thức tên file rồi tự đánh dấu đúng, dùng
clustering perceptual hash thay thế, hoặc random split theo patch.

Quy trình an toàn:

1. Đối chiếu README trong archive và tên file thực tế.
2. Điền đầy đủ `source_group` trong `group_map_template.csv`.
3. Đánh dấu `verified=true` chỉ sau khi review; giữ cột SHA-256 để phát hiện map
   cũ được áp dụng nhầm cho archive đã đổi.
4. Audit lại:

```powershell
python scripts/audit_data.py data/raw/SDNET2018 `
  --group-map data/manifests/group_map_verified.csv
```

Có thể dùng regex đã kiểm chứng thay cho map:

```powershell
python scripts/audit_data.py data/raw/SDNET2018 `
  --group-regex '...(?P<source_group>...)...' `
  --confirm-group-rule-verified
```

Cờ `--confirm-group-rule-verified` là lời xác nhận có chủ ý sau review, không
phải cơ chế tự phát hiện. Regex phải khớp mọi ảnh.

## Tạo và khóa split

Sau khi mọi ảnh hợp lệ và mọi nhóm được xác minh:

```powershell
python scripts/curate_manifest.py data/manifests/audit_manifest.csv `
  --output-dir data/manifests/pre_split_curation_v1

python scripts/create_splits.py `
  data/manifests/pre_split_curation_v1/pre_split_manifest.csv `
  --output-dir data/manifests/split_v1 `
  --conflict-report data/manifests/pre_split_curation_v1/conflict_report.json `
  --seed 42
```

Script dùng seed 42 và tỷ lệ mục tiêu 70% train, 15% validation, 15% test; cân
bằng theo tổ hợp bề mặt × nhãn. `source_group` là đơn vị bất khả phân. Các nhóm
có cùng exact SHA-256 cũng được nối thành một đơn vị, nên duplicate không thể
rò sang split khác. Script fail fast khi thiếu/không xác minh nhóm, có file lỗi,
hash trống, nhãn xung đột hoặc một stratum không đủ nhóm độc lập.

Locked `split_v1` hiện tại gồm 39.014 ảnh train, 8.540 ảnh validation và 8.534
ảnh test; tương ứng 160/35/35 `source_group`. Đầu ra gồm:

- `manifest.csv`, `train.csv`, `validation.csv`, `test.csv`;
- `split_audit.json`, bắt buộc `valid=true` và không overlap path/group/hash;
- `manifest_hashes.json`, chứa SHA-256 từng file, hash canonical, seed và tỷ lệ;
- `split_complete.json` và snapshot lineage của audit/curation.

Canonical SHA-256 hiện tại của `manifest.csv` là
`b38302d62547fb20f264b811f38f7031cf1b8d4c8a21f9db15d4dc39699a3da7`.

Các file này là bất biến. Script từ chối ghi đè. Sau khi đã nhìn kết quả test,
không xóa rồi tạo lại split. Nếu phát hiện lỗi dữ liệu, ghi nhận protocol bị hủy,
sửa từ nguồn và tạo phiên bản mới (`split_v2`) trước khi huấn luyện lại toàn bộ.

## Tiền xử lý và augmentation

`crackspot.data.build_tf_dataset` là data loader duy nhất. Nó:

- decode byte thật bằng Pillow, giới hạn mặc định 10 MB và 25 MP;
- chặn ảnh hỏng/decompression bomb, tôn trọng EXIF, đổi grayscale/RGBA sang RGB;
- resize 224×224 và gọi đúng MobileNetV2 `preprocess_input` (`[-1, 1]`);
- chỉ cho augmentation khi `training=True`: flip ngang 50%, xoay tối đa 15°,
  brightness/contrast nhẹ tối đa 20%; validation/test không augmentation.

Class weight cân bằng phải tính bằng
`compute_balanced_class_weights(train_frame)`, chỉ trên manifest train.

## Ảnh tự chụp

Đặt ảnh do nhóm tự chụp trong `data/external/real/` ở máy làm thí nghiệm và tạo
manifest/nhãn riêng. Không gộp chúng vào test SDNET2018, không commit ảnh có dữ
liệu riêng tư, và không dùng ảnh sinh tổng hợp rồi gọi là ảnh thực tế.

Manifest CSV tối thiểu dùng schema sau (mỗi đường dẫn phải duy nhất):

```text
relative_path,label,capture_source,capture_id,notes
```

- `label`: `1` = Crack, `0` = Non-crack;
- `capture_source`: ghi chính xác `self_captured` cho ảnh nhóm tự chụp;
- `capture_id`: mã ảnh gốc/cảnh chụp để truy vết, không phải split SDNET;
- không dùng giá trị `train`, `validation` hay `test` trong cột `split` nếu có.

Sau khi checkpoint và threshold đã khóa bằng validation:

```powershell
python scripts/evaluate_real_images.py `
  --selection artifacts/runs/E4_RUN_ID/selection_complete.json `
  --manifest data/external/real/manifest.csv `
  --dataset-root data/external/real `
  --output-dir artifacts/report/real_images/E4_RUN_ID `
  --confirm-self-captured
```

Output có `metrics_real.json`, `predictions_real.csv`, ma trận nhầm lẫn và môi
trường chạy, luôn ghi `included_in_standard_test=false`. Cờ xác nhận là cam kết
nguồn ảnh có chủ ý; công cụ không thể tự chứng minh quyền sở hữu hay hoàn cảnh chụp.
