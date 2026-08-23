from __future__ import annotations

import json

import pytest

from scripts import platform_doctor


def test_platform_doctor_validates_complete_checkout() -> None:
    checks = platform_doctor.run_checks()

    assert all(check.passed for check in checks)
    assert "Case-study video" in {check.name for check in checks}


def test_platform_doctor_emits_machine_readable_results(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert platform_doctor.main(["--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["checks"]
    assert all(check["passed"] for check in payload["checks"])
