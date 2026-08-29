# Quy ước artifacts

Artifacts được sinh từ chạy thật, không tạo file số liệu “mẫu” trong thư mục report.

## Phân vùng

```text
artifacts/
├── runs/<run_id>/        # run E1-E5 bất biến
├── smoke/<run_id>/       # tiny/synthetic, NOT_VALID_FOR_REPORT
├── verification/         # lint/test/smoke/demo logs
└── report/               # chỉ bằng chứng chính thức
```

## Mỗi run tối thiểu

- config snapshot/hash;
- manifest/split hash;
- environment, pip freeze, Git commit;
- seed/deterministic;
- log và `history.csv`;
- best `.keras` + SHA-256;
- validation metrics/predictions;
- classification report/confusion matrix/curves;
- total/trainable parameters;
- training duration và benchmark;
- FP/FN và danh sách ảnh phân tích.

Test metrics/predictions chỉ sinh bởi final evaluator sau selection lock.

## Report root

Tối thiểu gồm dataset summary, comparison E1-E5, confusion matrices, curves, threshold curve, Grad-CAM grid, latency table, `report_facts.json`, `comparison_table.md` và evidence map.

`report/<bundle_id>/report_facts.json` là nguồn duy nhất cho số trong báo cáo/slide; notebook chính mặc định dùng bundle `final_bundle_v1`. Mỗi fact phải truy được tới run/checkpoint/manifest hash.

## Bất biến và dọn dẹp

- Không ghi đè `run_id`; resume phải xác minh config/hash.
- Không copy smoke vào report.
- Không commit dataset/upload/log cá nhân/secret.
- Checkpoint chính thức được copy hoặc phát hành theo `models/README.md` cùng hash.
- Khi bundle, tạo SHA-256 cho toàn bộ bundle và ghi biên bản bàn giao.
