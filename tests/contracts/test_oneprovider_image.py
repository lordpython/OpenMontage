"""Contract tests for the oneprovider.dev image adapter.

All HTTP calls are mocked. This suite must never make a paid call.
"""

from __future__ import annotations

import base64

import pytest

from tools.base_tool import BaseTool, ToolRuntime, ToolStatus
from tools.graphics.oneprovider_image import OneProviderImage
from tools.provider_pricing import PriceQuoteRequired

# Shape of a real gpt-image-2 response (verified live 2026-10-11).
_USAGE = {
    "input_tokens": 11,
    "input_tokens_details": {"image_tokens": 0, "text_tokens": 11},
    "output_tokens": 2058,
    "output_tokens_details": {"image_tokens": 2058, "text_tokens": 0},
    "total_tokens": 2069,
}


class _FakeResponse:
    def __init__(
        self,
        payload: dict | None = None,
        *,
        content: bytes = b"",
        status_code: int = 200,
        text: str = "",
    ) -> None:
        self._payload = payload if payload is not None else {}
        self.content = content
        self.status_code = status_code
        self.text = text

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class TestContract:
    def test_identity_and_capabilities(self):
        assert issubclass(OneProviderImage, BaseTool)
        tool = OneProviderImage()
        assert tool.name == "oneprovider_image"
        assert tool.provider == "oneprovider"
        assert tool.capability == "image_generation"
        assert tool.runtime == ToolRuntime.API
        assert tool.supports["text_to_image"] is True
        assert tool.supports["image_edit"] is True
        assert "env:ONEPROVIDER_API_KEY" in tool.dependencies

    def test_status_requires_oneprovider_api_key(self, monkeypatch):
        monkeypatch.delenv("ONEPROVIDER_API_KEY", raising=False)
        assert OneProviderImage().get_status() == ToolStatus.UNAVAILABLE
        monkeypatch.setenv("ONEPROVIDER_API_KEY", "sk-fake-key")
        assert OneProviderImage().get_status() == ToolStatus.AVAILABLE

    def test_estimate_cost_requires_quote_and_dry_run_reports_it(self):
        tool = OneProviderImage()
        with pytest.raises(PriceQuoteRequired, match="response usage"):
            tool.estimate_cost({"prompt": "x"})
        dry = tool.dry_run({"prompt": "x"})
        assert dry["cost_status"] == "quote_required"
        assert dry["estimated_cost_usd"] is None


class TestGeneration:
    def test_generate_sends_only_documented_fields(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ONEPROVIDER_API_KEY", "sk-fake-key")
        captured = {}
        encoded = base64.b64encode(b"fake-png-bytes").decode("ascii")

        def fake_post(url, *, headers, json, timeout):
            captured.update(url=url, headers=headers, json=json, timeout=timeout)
            return _FakeResponse(
                {
                    "created": 1791677349,
                    "data": [
                        {
                            "b64_json": encoded,
                            "output_format": "png",
                            "revised_prompt": "A cobalt-blue ceramic teapot",
                            "size": "1024x1024",
                            "quality": "low",
                        }
                    ],
                    "usage": _USAGE,
                }
            )

        monkeypatch.setattr("requests.post", fake_post)
        output = tmp_path / "teapot.png"
        result = OneProviderImage().execute(
            {"prompt": "A cobalt-blue ceramic teapot", "output_path": str(output)}
        )

        assert result.success is True
        assert output.read_bytes() == b"fake-png-bytes"
        assert captured["url"] == (
            "https://api.oneprovider.dev/v1/images/generations"
        )
        assert captured["headers"]["Authorization"] == "Bearer sk-fake-key"
        # Exact wire payload: defaults only, no undocumented injections.
        assert captured["json"] == {
            "model": "gpt-image-2",
            "prompt": "A cobalt-blue ceramic teapot",
            "size": "1024x1024",
            "quality": "low",
        }
        assert result.data["operation"] == "generate"
        assert result.data["revised_prompt"] == "A cobalt-blue ceramic teapot"
        assert result.data["cost_status"] == "computed_from_response_usage"
        assert result.data["usage"] == _USAGE
        # 11 text input tokens at $8/1M + 2058 image output tokens at $30/1M.
        assert result.cost_usd == pytest.approx(0.061828)
        assert result.artifacts == [str(output)]

    def test_optional_fields_pass_through_when_provided(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ONEPROVIDER_API_KEY", "sk-fake-key")
        captured = {}

        def fake_post(url, *, headers, json, timeout):
            captured["json"] = json
            return _FakeResponse(
                {
                    "data": [{"b64_json": base64.b64encode(b"png").decode("ascii")}],
                    "usage": _USAGE,
                }
            )

        monkeypatch.setattr("requests.post", fake_post)
        result = OneProviderImage().execute(
            {
                "prompt": "x",
                "model": "gpt-image-2.5-sunburst",
                "size": "1536x1024",
                "quality": "high",
                "output_format": "webp",
                "output_path": str(tmp_path / "out.img"),
            }
        )
        assert result.success is True
        assert captured["json"] == {
            "model": "gpt-image-2.5-sunburst",
            "prompt": "x",
            "size": "1536x1024",
            "quality": "high",
            "output_format": "webp",
        }
        # Requested suffix wins over the response format.
        assert (tmp_path / "out.img").read_bytes() == b"png"

    def test_url_payload_is_downloaded(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ONEPROVIDER_API_KEY", "sk-fake-key")
        monkeypatch.setattr(
            "requests.post",
            lambda *args, **kwargs: _FakeResponse(
                {
                    "data": [{"url": "https://cdn.example/teapot.png"}],
                    "usage": _USAGE,
                }
            ),
        )
        monkeypatch.setattr(
            "requests.get",
            lambda *args, **kwargs: _FakeResponse(content=b"downloaded-bytes"),
        )
        output = tmp_path / "dl.png"
        result = OneProviderImage().execute(
            {"prompt": "x", "output_path": str(output)}
        )
        assert result.success is True
        assert output.read_bytes() == b"downloaded-bytes"

    def test_usage_missing_reports_cost_unknown_not_zero(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ONEPROVIDER_API_KEY", "sk-fake-key")
        monkeypatch.setattr(
            "requests.post",
            lambda *args, **kwargs: _FakeResponse(
                {"data": [{"b64_json": base64.b64encode(b"png").decode("ascii")}]}
            ),
        )
        result = OneProviderImage().execute(
            {"prompt": "x", "output_path": str(tmp_path / "a.png")}
        )
        assert result.success is True
        assert result.cost_usd is None
        assert result.data["cost_status"] == "usage_missing"


class TestEdits:
    def test_edit_uses_multipart_route_with_documented_fields(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("ONEPROVIDER_API_KEY", "sk-fake-key")
        captured = {}
        source = tmp_path / "source.png"
        source.write_bytes(b"source-image")

        def fake_post(url, *, headers, data, files, timeout):
            captured.update(url=url, headers=headers, data=data, files=files)
            return _FakeResponse(
                {
                    "data": [{"b64_json": base64.b64encode(b"edited").decode("ascii")}],
                    "usage": _USAGE,
                }
            )

        monkeypatch.setattr("requests.post", fake_post)
        output = tmp_path / "edited.png"
        result = OneProviderImage().execute(
            {
                "prompt": "remove the handle",
                "image_path": str(source),
                "output_path": str(output),
            }
        )
        assert result.success is True
        assert output.read_bytes() == b"edited"
        assert captured["url"] == "https://api.oneprovider.dev/v1/images/edits"
        assert captured["data"] == {
            "model": "gpt-image-2",
            "prompt": "remove the handle",
        }
        assert "image" in captured["files"]


class TestInputSafety:
    def test_missing_key_never_reaches_network(self, monkeypatch):
        monkeypatch.delenv("ONEPROVIDER_API_KEY", raising=False)

        def fail_network(*args, **kwargs):
            raise AssertionError("missing key must not call the network")

        monkeypatch.setattr("requests.post", fail_network)
        result = OneProviderImage().execute({"prompt": "x"})
        assert result.success is False
        assert "ONEPROVIDER_API_KEY" in result.error

    def test_invalid_input_never_reaches_network(self, monkeypatch):
        monkeypatch.setenv("ONEPROVIDER_API_KEY", "sk-fake-key")

        def fail_network(*args, **kwargs):
            raise AssertionError("invalid input must not call the network")

        monkeypatch.setattr("requests.post", fail_network)
        result = OneProviderImage().execute({})
        assert result.success is False
        assert "prompt" in result.error

    def test_non_https_base_url_rejected_before_network(self, monkeypatch):
        monkeypatch.setenv("ONEPROVIDER_API_KEY", "sk-fake-key")
        monkeypatch.setenv("ONEPROVIDER_BASE_URL", "http://insecure.example/v1")

        def fail_network(*args, **kwargs):
            raise AssertionError("non-https base URL must not call the network")

        monkeypatch.setattr("requests.post", fail_network)
        result = OneProviderImage().execute({"prompt": "x"})
        assert result.success is False
        assert "https" in result.error

    def test_http_error_redacts_key(self, monkeypatch):
        api_key = "sk-unit-test-redaction-marker"
        monkeypatch.setenv("ONEPROVIDER_API_KEY", api_key)
        monkeypatch.setattr(
            "requests.post",
            lambda *args, **kwargs: _FakeResponse(
                status_code=429,
                text=(
                    '{"error":{"type":"insufficient_balance",'
                    f'"message":"Insufficient balance on this API key ({api_key})"}}'
                ),
            ),
        )
        result = OneProviderImage().execute({"prompt": "x"})
        assert result.success is False
        assert "429" in result.error
        assert "insufficient_balance" in result.error
        assert api_key not in result.error
        assert "[redacted]" in result.error

    def test_exception_redacts_key(self, monkeypatch):
        api_key = "sk-unit-test-exception-marker"
        monkeypatch.setenv("ONEPROVIDER_API_KEY", api_key)

        def failing_post(*args, **kwargs):
            raise RuntimeError(f"connection reset while using {api_key}")

        monkeypatch.setattr("requests.post", failing_post)
        result = OneProviderImage().execute({"prompt": "x"})
        assert result.success is False
        assert api_key not in result.error
        assert "[redacted]" in result.error
