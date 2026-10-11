"""oneprovider.dev OpenAI-Images-compatible generation and edits.

Gateway docs: https://oneprovider.dev/docs/llms.txt (§1 endpoints, §4.7 images).
Only the documented request fields are sent — the gateway rejects unknown
injections on routed models, so optional fields pass through only when the
caller supplies them.
"""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Any

from tools.provider_pricing import PriceQuoteRequired

from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ResourceProfile,
    RetryPolicy,
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolTier,
)

DEFAULT_BASE_URL = "https://api.oneprovider.dev/v1"

# Documented image token rates (docs §4.7), USD per 1M tokens.
_IMAGE_INPUT_USD_PER_M = 8.0
_CACHED_IMAGE_INPUT_USD_PER_M = 2.0
_IMAGE_OUTPUT_USD_PER_M = 30.0


class OneProviderImage(BaseTool):
    name = "oneprovider_image"
    version = "0.1.0"
    tier = ToolTier.GENERATE
    capability = "image_generation"
    provider = "oneprovider"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.STOCHASTIC
    runtime = ToolRuntime.API

    dependencies = ["env:ONEPROVIDER_API_KEY"]
    install_instructions = (
        "Set ONEPROVIDER_API_KEY to your oneprovider.dev API key.\n"
        "  Issue one in the cabinet: @oneprovider_robot on Telegram or\n"
        "  https://app.oneprovider.dev\n"
        "  Optional: ONEPROVIDER_BASE_URL (default https://api.oneprovider.dev/v1)"
    )
    agent_skills = ["provider-model-refresh"]  # GPT Image 2.5 usage guidance

    capabilities = ["generate_image", "edit_image", "text_to_image"]
    supports = {
        "text_to_image": True,
        "image_edit": True,
        "multiple_outputs": False,
        "aspect_ratio": True,
        "resolution": True,
    }
    best_for = [
        "GPT Image 2 generations and edits through the oneprovider.dev gateway",
        "one-key access to gemini-3-pro-image and grok-imagine image models",
        "low-cost drafts at quality=low (1024x1024 is about $0.06)",
    ]
    not_good_for = ["offline generation", "strict seeded reproducibility"]

    input_schema = {
        "type": "object",
        "required": ["prompt"],
        "properties": {
            "prompt": {"type": "string"},
            "model": {
                "type": "string",
                "enum": [
                    "gpt-image-2",
                    "gpt-image-2.5-sunburst",
                    "gemini-3-pro-image",
                    "gemini-3.1-flash-image",
                    "grok-imagine-image-2.0",
                ],
                "default": "gpt-image-2",
            },
            "size": {
                "type": "string",
                "enum": ["1024x1024", "1536x1024", "1024x1536", "auto"],
                "default": "1024x1024",
            },
            "quality": {
                "type": "string",
                "enum": ["low", "medium", "high", "auto"],
                "default": "low",
                "description": "Cost scales with output tokens; low is the safe draft setting.",
            },
            "output_format": {
                "type": "string",
                "enum": ["png", "jpeg", "webp"],
                "description": "Only sent when provided; not every routed model accepts it.",
            },
            "image_path": {
                "type": "string",
                "description": "Source image; switches to POST /images/edits (multipart).",
            },
            "output_path": {"type": "string"},
        },
    }

    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=512, vram_mb=0, disk_mb=100, network_required=True
    )
    retry_policy = RetryPolicy(
        max_retries=2,
        retryable_errors=["rate_limit", "server_error", "timeout"],
    )
    idempotency_key_fields = [
        "prompt",
        "model",
        "size",
        "quality",
        "output_format",
        "image_path",
    ]
    side_effects = ["writes image file to output_path", "calls the oneprovider.dev API"]
    user_visible_verification = ["Inspect the generated image for relevance and quality"]

    def _get_base_url(self) -> str:
        base_url = os.environ.get("ONEPROVIDER_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
        if not base_url.startswith("https://"):
            raise ValueError("ONEPROVIDER_BASE_URL must be an https:// URL")
        return base_url

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        # Image token counts are only known from the response usage (docs §4.7):
        # never present a guessed number as an estimate.
        raise PriceQuoteRequired(
            "oneprovider.dev bills image models per token; cost is computed from "
            "the response usage after generation"
        )

    @staticmethod
    def _cost_from_usage(usage: dict[str, Any] | None) -> float | None:
        """Compute billed cost from the response usage (docs §4.7).

        Image input $8/1M, cached image input $2/1M, image output $30/1M.
        Text tokens have no separately published rate on this route; they are
        priced at the image rate so the figure is never silently zero.
        """
        if not usage:
            return None
        in_details = usage.get("input_tokens_details") or {}
        out_details = usage.get("output_tokens_details") or {}
        image_in = in_details.get("image_tokens", usage.get("input_tokens", 0)) or 0
        text_in = in_details.get("text_tokens", 0) or 0
        cached = in_details.get("cached_tokens", 0) or 0
        image_out = out_details.get("image_tokens", usage.get("output_tokens", 0)) or 0
        text_out = out_details.get("text_tokens", 0) or 0
        cost = (
            (image_in + text_in) * _IMAGE_INPUT_USD_PER_M
            + cached * _CACHED_IMAGE_INPUT_USD_PER_M
            + (image_out + text_out) * _IMAGE_OUTPUT_USD_PER_M
        ) / 1_000_000
        return round(cost, 6)

    @staticmethod
    def _output_paths(
        output_path: str | None, count: int, extension: str
    ) -> list[Path]:
        """Derive one output path per returned image.

        With a single image, honor the requested path as-is. With several,
        suffix each with `_1`, `_2`, ... so no image overwrites another.
        """
        ext = extension if extension.startswith(".") else f".{extension}"
        if not output_path:
            return [Path(f"generated_image_{idx + 1}{ext}") for idx in range(count)]

        path = Path(output_path)
        suffix = path.suffix or ext
        if count == 1:
            return [path if path.suffix else path.with_suffix(suffix)]

        base = path.with_suffix("") if path.suffix else path
        return [base.parent / f"{base.name}_{idx + 1}{suffix}" for idx in range(count)]

    @staticmethod
    def _redact(text: str, api_key: str) -> str:
        return text.replace(api_key, "[redacted]") if api_key else text

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        api_key = os.environ.get("ONEPROVIDER_API_KEY")
        if not api_key:
            return ToolResult(
                success=False,
                error="ONEPROVIDER_API_KEY not set. " + self.install_instructions,
            )

        import requests

        start = time.time()
        try:
            from jsonschema import validate

            validate(inputs, self.input_schema)
            base_url = self._get_base_url()
            model = inputs.get("model", "gpt-image-2")
            image_path = inputs.get("image_path")

            headers = {"Authorization": f"Bearer {api_key}"}
            if image_path:
                # Documented edit route: multipart with model, prompt and one image.
                with open(image_path, "rb") as image_file:
                    response = requests.post(
                        f"{base_url}/images/edits",
                        headers=headers,
                        data={"model": model, "prompt": inputs["prompt"]},
                        files={"image": image_file},
                        timeout=360,
                    )
                operation = "edit"
            else:
                payload: dict[str, Any] = {
                    "model": model,
                    "prompt": inputs["prompt"],
                    "size": inputs.get("size", "1024x1024"),
                    "quality": inputs.get("quality", "low"),
                }
                if inputs.get("output_format"):
                    payload["output_format"] = inputs["output_format"]
                response = requests.post(
                    f"{base_url}/images/generations",
                    headers={**headers, "Content-Type": "application/json"},
                    json=payload,
                    timeout=360,
                )
                operation = "generate"

            if response.status_code >= 400:
                return ToolResult(
                    success=False,
                    error=self._redact(
                        f"oneprovider.dev {operation} failed (HTTP "
                        f"{response.status_code}): {response.text[:400]}",
                        api_key,
                    ),
                )
            body = response.json()

            items = body.get("data") or []
            if not items:
                return ToolResult(
                    success=False, error="oneprovider.dev returned no image outputs"
                )

            usage = body.get("usage")
            cost = self._cost_from_usage(usage)
            extension = str(items[0].get("output_format") or "png")
            output_paths = self._output_paths(
                inputs.get("output_path"), len(items), extension
            )
            outputs: list[str] = []
            for item, out_path in zip(items, output_paths):
                out_path.parent.mkdir(parents=True, exist_ok=True)
                if item.get("b64_json"):
                    out_path.write_bytes(base64.b64decode(item["b64_json"]))
                elif item.get("url"):
                    download = requests.get(item["url"], timeout=120)
                    download.raise_for_status()
                    out_path.write_bytes(download.content)
                else:
                    return ToolResult(
                        success=False,
                        error="oneprovider.dev image output missing b64_json/url",
                    )
                outputs.append(str(out_path))

        except Exception as exc:
            return ToolResult(
                success=False,
                error=self._redact(
                    f"oneprovider.dev image generation failed: {exc}", api_key
                ),
            )

        return ToolResult(
            success=True,
            data={
                "provider": self.provider,
                "model": model,
                "prompt": inputs["prompt"],
                "operation": operation,
                "revised_prompt": items[0].get("revised_prompt"),
                "output": outputs[0],
                "outputs": outputs,
                "images_generated": len(outputs),
                "usage": usage,
                "cost_status": "computed_from_response_usage"
                if cost is not None
                else "usage_missing",
            },
            artifacts=outputs,
            cost_usd=cost,
            duration_seconds=round(time.time() - start, 2),
            model=model,
        )
