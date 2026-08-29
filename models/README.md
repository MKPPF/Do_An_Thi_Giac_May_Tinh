# Checkpoint CrackSpot

Không commit checkpoint thử nghiệm lớn vào Git thường. Model chính thức gồm:

```text
models/crackspot.keras
models/crackspot.metadata.json
```

Metadata cần có run/model version, threshold, input size, preprocessing, label mapping, Grad-CAM layer, TensorFlow version, model SHA-256 và manifest SHA-256. Schema được mô tả trong `MODEL_CARD.md`.

## Cung cấp model

Ưu tiên một trong hai cách:

1. Git LFS cho `.keras`, chỉ khi quota/repository phù hợp;
2. release/bundle riêng có URL và SHA-256 được công bố trong README bàn giao.

Không đặt link giả hoặc file checkpoint rỗng. Nếu model chưa có, app phải hiển thị trạng thái thiếu model rõ ràng.

## Xác minh Windows

```powershell
Get-FileHash .\models\crackspot.keras -Algorithm SHA256
Get-Content .\models\crackspot.metadata.json
```

Hash phải khớp metadata/selection lock. Không dùng model nếu hash sai hoặc metadata không đúng label/preprocessing.
