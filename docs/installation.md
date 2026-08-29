# Cài đặt và môi trường

## Bộ phiên bản mục tiêu

- Python 3.12;
- TensorFlow 2.19.0;
- Windows native: CPU;
- GPU: Google Colab hoặc WSL2.

Đề cương ưu tiên TensorFlow 2.15.x trên Python 3.10/3.11. Máy triển khai thực tế có Python 3.12.6, nên repository khóa TensorFlow 2.19.0 và bộ dependency tương thích đã chạy test thay vì tuyên bố sai môi trường 2.15.x. Smoke hiện tại ghi Windows 11, TensorFlow 2.19.0 và CPU trong `environment.json`; mọi run chính thức phải ghi lại môi trường của chính nó.

TensorFlow 2.19 không cung cấp GPU native Windows theo lộ trình hỗ trợ hiện hành; không cài CUDA tùy ý rồi tuyên bố Windows native GPU hoạt động. Môi trường mỗi run phải ghi nhận thiết bị thật.

## Windows PowerShell

```powershell
cd D:\Đồ án\CrackSpot
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pip install -e .
python -m ipykernel install --user --name crackspot --display-name "Python (CrackSpot)"
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices())"
.\.venv\Scripts\jupyter-lab.exe CrackSpot.ipynb
```

Nếu PowerShell chặn activate, có thể dùng trực tiếp:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

## WSL2/Linux

Nếu WSL2 chưa được cài, mở PowerShell bằng quyền Administrator, chạy `wsl --install -d Ubuntu`, khởi động lại Windows và hoàn tất tạo user Ubuntu. Trong Ubuntu:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pip install "tensorflow[and-cuda]==2.19.0"
python -m pip install -e .
python -m ipykernel install --user --name crackspot-wsl --display-name "Python (CrackSpot WSL GPU)"
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices())"
```

GPU WSL2 cần driver NVIDIA phía Windows và môi trường tương thích TensorFlow. Không pin/cài một bộ CUDA riêng nếu TensorFlow pip đang quản lý dependency khác; ghi lại output thiết bị trước khi train.

## Google Colab

Mở `CrackSpot.ipynb`, chọn GPU nếu có; URL repository và chế độ clone/cài dependency tự động đã được cấu hình sẵn. Bật `DO_DOWNLOAD_SDNET2018`, rồi chạy từ trên xuống. Notebook clone locked manifests vào `/content/CrackSpot`, nhưng toàn bộ code thực thi nằm trực tiếp trong notebook, không phụ thuộc package `crackspot` hay `scripts/`. Sau đó notebook tải, xác minh dataset thật và ghi artifact vào `artifacts/notebook/`. Có thể sao lưu artifact sang Drive sau run; không nên train trực tiếp trên cây nhiều file ảnh trong Drive vì I/O ngẫu nhiên chậm.

Kiểm tra phiên bản:

```bash
python --version
python -m pip check
python -c "import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices())"
```

## Nhóm dependency

- `requirements.txt`: runtime demo/inference.
- `requirements-train.txt`: runtime + audit/train/evaluate/report.
- `requirements-dev.txt`: training + pytest/Ruff.

Ba file dùng include để không tạo ba bộ pin mâu thuẫn. `pyproject.toml` là cấu hình package do dự án quản lý; tài liệu này không thay nó.

## Kiểm tra sạch

```powershell
python -m pip check
ruff check .
ruff format --check .
pytest -q
```

Lưu output, thời gian và commit trong `artifacts/verification/`. `pytest` phải chạy khi không có network, dataset và checkpoint thật.

## Lỗi thường gặp

### Không import được package

Chạy từ root và cài editable:

```powershell
python -m pip install -e .
```

### TensorFlow không thấy GPU

- Windows native với TF 2.19: đây là kỳ vọng CPU; dùng WSL2/Colab.
- Colab: chọn Runtime/Change runtime type/GPU, rồi restart runtime sau khi cài dependency nếu cần.
- WSL2: kiểm tra `nvidia-smi` và `tf.config.list_physical_devices('GPU')`.

### Hết bộ nhớ

Giảm `pipeline.batch_size` trong bản config run mới, không sửa snapshot của run cũ. Ghi rõ thay đổi dựa trên tài nguyên, không dựa vào test.

### Streamlit báo thiếu model

Đặt `models/crackspot.keras` và `models/crackspot.metadata.json`, hoặc khai báo `CRACKSPOT_MODEL_PATH` và `CRACKSPOT_METADATA_PATH`. Xác minh hash trước khi chạy.

### Full split bị từ chối

Đây là hành vi đúng nếu `group_rule_verified: false`, có dòng chưa xác minh nhóm, hoặc cùng exact hash mang nhãn mâu thuẫn. Với dữ liệu chưa kiểm chứng, xác minh quy tắc source group trên archive/README thật hoặc cung cấp `group_map.csv`; không tắt cổng chỉ để train nhanh.

Với audit SDNET2018 native hiện tại, source group đã được xác minh và hai exact hash/bốn file có nhãn mâu thuẫn đã được xử lý trong `data/manifests/pre_split_curation_v1/`. Phải dùng `pre_split_manifest.csv` đã khóa làm input split; không dùng lại `audit_manifest.csv`, không sửa nhãn và không tái sinh/ghi đè bundle curation. Smoke không group phải gắn `NOT_VALID_FOR_REPORT`.
