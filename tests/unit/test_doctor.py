from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

import pytest

from scripts import doctor


class _CudaStub:
    def __init__(self, *, available: bool, count: int = 0) -> None:
        self._available = available
        self._count = count

    def is_available(self) -> bool:
        return self._available

    def device_count(self) -> int:
        return self._count

    def get_device_name(self, index: int) -> str:
        return f"GPU {index}"


class _MpsStub:
    def __init__(self, *, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


def _torch_stub(*, cuda_available: bool = False, cuda_count: int = 0, mps: bool = False) -> Any:
    return SimpleNamespace(
        cuda=_CudaStub(available=cuda_available, count=cuda_count),
        backends=SimpleNamespace(mps=_MpsStub(available=mps)),
    )


def test_device_check_validates_requested_cuda_index(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NPR_DEVICE", "cuda:1")
    monkeypatch.setattr(
        importlib, "import_module", lambda _: _torch_stub(cuda_available=True, cuda_count=1)
    )

    result = doctor._device_check()

    assert result.status == "FAIL"
    assert "only 1 available" in result.message


def test_device_check_reports_selected_cuda_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NPR_DEVICE", "cuda:1")
    monkeypatch.setattr(
        importlib, "import_module", lambda _: _torch_stub(cuda_available=True, cuda_count=2)
    )

    result = doctor._device_check()

    assert result.status == "PASS"
    assert result.message == "cuda:1 (GPU 1)"


def test_device_check_rejects_unavailable_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NPR_DEVICE", "mps")
    monkeypatch.setattr(importlib, "import_module", lambda _: _torch_stub())

    result = doctor._device_check()

    assert result.status == "FAIL"
    assert result.message == "MPS requested but unavailable"
