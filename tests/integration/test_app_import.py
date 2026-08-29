from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


def load_app_module() -> Any:
    app_path = Path(__file__).resolve().parents[2] / "app.py"
    spec = importlib.util.spec_from_file_location("crackspot_streamlit_app_test", app_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MissingModelUI:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.infos: list[str] = []

    def set_page_config(self, **_: Any) -> None:
        pass

    def title(self, *_: Any, **__: Any) -> None:
        pass

    def write(self, *_: Any, **__: Any) -> None:
        pass

    def warning(self, *_: Any, **__: Any) -> None:
        pass

    def error(self, message: str, *_: Any, **__: Any) -> None:
        self.errors.append(message)

    def caption(self, *_: Any, **__: Any) -> None:
        pass

    def file_uploader(self, *_: Any, **__: Any) -> None:
        return None

    def info(self, message: str, *_: Any, **__: Any) -> None:
        self.infos.append(message)


def test_app_import_has_no_render_side_effect() -> None:
    module = load_app_module()

    assert callable(module.main)
    assert callable(module.render_app)


def test_app_missing_model_is_friendly(monkeypatch: Any, tmp_path: Path) -> None:
    module = load_app_module()
    monkeypatch.setenv("CRACKSPOT_MODEL_PATH", str(tmp_path / "missing.keras"))
    monkeypatch.delenv("CRACKSPOT_METADATA_PATH", raising=False)
    ui = MissingModelUI()

    module.render_app(ui)

    assert ui.errors
    assert "Chưa thể tải mô hình" in ui.errors[0]
    assert ui.infos
    assert "không được lưu" in ui.infos[0]
