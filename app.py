"""CrackSpot Streamlit demo.

Run from the repository root after installing the demo dependencies:
``streamlit run app.py``.  Importing this module remains safe when Streamlit is
not installed, which keeps service-layer tests lightweight.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

try:
    import streamlit as st
except ImportError:  # Streamlit is an optional dependency outside the demo.
    st = None

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "crackspot.keras"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "models" / "crackspot.metadata.json"


def _configured_paths() -> tuple[Path, Path | None]:
    model_path = Path(os.environ.get("CRACKSPOT_MODEL_PATH", DEFAULT_MODEL_PATH)).expanduser()
    metadata_setting = os.environ.get("CRACKSPOT_METADATA_PATH")
    if metadata_setting:
        metadata_path: Path | None = Path(metadata_setting).expanduser()
    elif DEFAULT_METADATA_PATH.is_file() and model_path == DEFAULT_MODEL_PATH:
        metadata_path = DEFAULT_METADATA_PATH
    else:
        metadata_path = None
    return model_path.resolve(), metadata_path.resolve() if metadata_path else None


def _load_service_uncached(model_path: str, metadata_path: str | None) -> Any:
    from crackspot.inference.service import InferenceService

    return InferenceService.from_files(model_path, metadata_path)


if st is not None:
    load_service = st.cache_resource(show_spinner=False)(_load_service_uncached)
else:
    load_service = _load_service_uncached


def _png_bytes(image: Any) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def render_app(streamlit_module: Any | None = None) -> None:
    """Render the Vietnamese single-image demo; no upload is persisted."""

    ui = streamlit_module or st
    if ui is None:
        raise RuntimeError(
            "Chưa cài Streamlit. Hãy cài dependency demo rồi chạy: streamlit run app.py"
        )

    ui.set_page_config(
        page_title="CrackSpot - Phát hiện vết nứt",
        page_icon="🔎",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    ui.title("CrackSpot - Phát hiện vết nứt bề mặt")
    ui.write(
        "Tải một ảnh tường, đường, bê tông hoặc cầu. MobileNetV2 phân loại "
        "**Crack / Non-crack** và Grad-CAM cho biết vùng làm tăng xác suất vết nứt."
    )
    ui.warning(
        "Kết quả chỉ hỗ trợ khảo sát sơ bộ, không thay thế đánh giá của kỹ sư "
        "xây dựng và không kết luận mức độ nguy hiểm."
    )

    model_path, metadata_path = _configured_paths()
    service = None
    model_error: Exception | None = None
    try:
        service = load_service(
            str(model_path), str(metadata_path) if metadata_path is not None else None
        )
    except Exception as exc:  # Display a friendly page for absent/corrupt deployment assets.
        model_error = exc
        ui.error(
            "Chưa thể tải mô hình. Demo vẫn mở để kiểm tra giao diện, nhưng nút dự đoán sẽ bị khóa."
        )
        ui.caption(f"Chi tiết: {exc}")

    uploaded = ui.file_uploader(
        "Kéo-thả hoặc chọn một ảnh",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
        help="JPEG/PNG, tối đa 10 MB và 25 megapixel. Tệp chỉ được xử lý trong bộ nhớ.",
    )
    if uploaded is None:
        ui.info("Chọn ảnh để bắt đầu. Ảnh tải lên không được lưu vào cơ sở dữ liệu.")
        return

    from crackspot.inference.preprocessing import ImageValidationError, decode_image

    payload = uploaded.getvalue()
    try:
        preview = decode_image(payload)
    except ImageValidationError as exc:
        ui.error(f"Ảnh không hợp lệ: {exc}")
        return

    ui.image(
        preview,
        caption="Ảnh đã kiểm tra và chuẩn hóa hướng EXIF",
        use_container_width=True,
    )
    detect = ui.button(
        "Phát hiện vết nứt",
        type="primary",
        disabled=service is None,
        use_container_width=True,
    )
    if service is None:
        if model_error is not None:
            ui.info(
                "Cần cung cấp `models/crackspot.keras` và metadata JSON tương ứng, "
                "hoặc đặt `CRACKSPOT_MODEL_PATH` / `CRACKSPOT_METADATA_PATH`."
            )
        return
    if not detect:
        return

    from crackspot.inference.service import InferenceError
    from crackspot.modeling.gradcam import GradCAMError

    try:
        with ui.spinner("Đang phân tích ảnh và tạo Grad-CAM..."):
            result = service.predict_image(payload, include_gradcam=True)
    except GradCAMError as exc:
        ui.warning(f"Không tạo được Grad-CAM: {exc}")
        try:
            result = service.predict_image(payload, include_gradcam=False)
        except InferenceError as prediction_exc:
            ui.error(f"Không thể dự đoán: {prediction_exc}")
            return
    except (ImageValidationError, InferenceError) as exc:
        ui.error(f"Không thể dự đoán: {exc}")
        return

    if result.is_crack:
        ui.error(f"Kết quả: **{result.display_label_vi}**")
    else:
        ui.success(f"Kết quả: **{result.display_label_vi}**")

    metric_columns = ui.columns(4)
    metric_columns[0].metric("Xác suất Crack", f"{result.crack_probability:.2%}")
    metric_columns[1].metric("Ngưỡng", f"{result.threshold:.3f}")
    metric_columns[2].metric("Suy luận", f"{result.latency_ms:.1f} ms")
    metric_columns[3].metric("Phiên bản model", result.model_version)
    ui.caption(f"Run ID: {result.run_id}")

    image_columns = ui.columns(2)
    image_columns[0].image(
        result.original_image,
        caption="Ảnh gốc đã xoay theo EXIF",
        use_container_width=True,
    )
    if result.overlay is not None:
        image_columns[1].image(
            result.overlay,
            caption="Grad-CAM - vùng làm tăng score Crack",
            use_container_width=True,
        )
        ui.download_button(
            "Tải ảnh Grad-CAM overlay",
            data=_png_bytes(result.overlay),
            file_name="crackspot_gradcam.png",
            mime="image/png",
        )

    ui.info(
        "Grad-CAM là bản đồ nhiệt vùng mô hình chú ý khi tính **P(Crack)**; "
        "không phải mask phân đoạn, bounding box hay định vị chính xác theo pixel. "
        "Ngay cả khi kết quả là Non-crack, màu nóng vẫn biểu diễn vùng kích hoạt score Crack, "
        "không phải mask của lớp Non-crack."
    )


def main() -> None:
    render_app()


if __name__ == "__main__":
    main()
