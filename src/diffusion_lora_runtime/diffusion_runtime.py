from __future__ import annotations

import base64
import gc
import io
import logging
import inspect
import random
import re
import threading
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from diffusers import (
    AutoPipelineForText2Image,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
)
from PIL import Image

from .checkpoint_inspection import detect_single_file_architecture
from .prompt_embedding import (
    PromptEmbeddingReference,
    PromptEmbeddingResolutionError,
    load_prompt_embedding_file,
    resolve_prompt_embedding_vectors,
)
from .schemas import LoRASpec, ModelSpec, StepCallback, StepProgress


SAMPLER_NAME = "euler_ancestral"
SCHEDULER_NAME = "simple"
MODEL_CPU_OFFLOAD = True
LORA_WEIGHT = 1.0
ILLUSTRIOUS_USE_SAFETENSORS = True
ILLUSTRIOUS_FORCE_V_PREDICTION_BY_NAME = True
ILLUSTRIOUS_V_PRED_MARKERS = ("vpred", "v-pred", "v_prediction", "v-prediction")

logger = logging.getLogger(__name__)


class AdapterLoadError(RuntimeError):
    pass


class GenerationCancelled(RuntimeError):
    pass


@dataclass
class LoadedAdapter:
    spec: LoRASpec
    device: str = "cpu"


@dataclass
class LoadedTextualInversion:
    token: str
    path: str
    tokens: list[str]


class DiffusionRuntime:
    def __init__(self, *, device: str = "auto", allow_cpu: bool = False) -> None:
        self.pipe: Any | None = None
        self.current_model: ModelSpec | None = None
        self.loaded_adapters: dict[str, LoadedAdapter] = {}
        self.loaded_textual_inversions: dict[str, LoadedTextualInversion] = {}
        self.active_adapter_signature: (
            tuple[tuple[str, ...], tuple[float, ...]] | None
        ) = None
        self._cancel_event = threading.Event()
        self._operation_lock = threading.RLock()
        self.requested_device = device
        self.allow_cpu = allow_cpu
        self.device = self._resolve_device(device)
        self.sampler_name = SAMPLER_NAME
        self.scheduler_name = SCHEDULER_NAME
        self._check_valid()

    @contextmanager
    def serialized_operation(self):
        with self._operation_lock:
            yield

    def _torch_dtype(self, dtype_name: str) -> torch.dtype:
        return {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype_name]

    def _resolve_device(self, requested_device: str) -> str:
        if requested_device != "auto":
            return requested_device
        if torch.cuda.is_available():
            return "cuda"
        mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
        if mps_backend is not None and callable(
            getattr(mps_backend, "is_available", None)
        ):
            try:
                if mps_backend.is_available():
                    return "mps"
            except Exception:
                pass
        return "cpu"

    def _device_type(self) -> str:
        return str(self.device).split(":", 1)[0]

    def _accelerator_diagnostics(self) -> str:
        cuda_version = getattr(torch.version, "cuda", None)
        torch_version = getattr(torch, "__version__", "unknown")
        return (
            f"torch={torch_version}, torch.version.cuda={cuda_version!r}, "
            f"torch.cuda.is_available()={torch.cuda.is_available()}, selected_device={self.device!r}"
        )

    def _check_valid(self):
        if self._device_type() == "cpu":
            if not self.allow_cpu:
                raise RuntimeError(
                    "No GPU/accelerator is available for diffusion inference. "
                    "The runtime selected CPU, but CPU inference is disabled by default because SDXL/FP16 CPU fallback "
                    "usually appears to hang. Install a CUDA-enabled PyTorch build in this environment, or explicitly use "
                    "DiffusionGenerator(device='cpu', allow_cpu=True) with torch_dtype='float32' for CPU debugging. "
                    f"Diagnostics: {self._accelerator_diagnostics()}"
                )

    def _effective_torch_dtype(self, model: ModelSpec) -> torch.dtype:
        requested_dtype = self._torch_dtype(model.torch_dtype)
        if self._device_type() == "cpu":
            self._check_valid()
            if requested_dtype is not torch.float32:
                logger.warning(
                    "CPU inference was explicitly enabled; overriding model torch_dtype=%s to float32.",
                    model.torch_dtype,
                )
                return torch.float32
        return requested_dtype

    def configure_ksampler(
        self, *, sampler_name: str = SAMPLER_NAME, scheduler_name: str = SCHEDULER_NAME
    ) -> None:
        """Remember the requested KSample sampler metadata before loading a model."""
        supported_samplers = {"euler_ancestral", "euler"}
        if sampler_name not in supported_samplers:
            raise ValueError(
                f"Unsupported sampler_name {sampler_name!r}; expected one of {sorted(supported_samplers)}."
            )
        self.sampler_name = sampler_name
        self.scheduler_name = scheduler_name

    def _looks_like_v_prediction_checkpoint(self, model_path: Path) -> bool:
        normalized_name = model_path.name.casefold()
        return any(marker in normalized_name for marker in ILLUSTRIOUS_V_PRED_MARKERS)

    def _single_file_pipeline_class(self, model_path: Path) -> type[Any]:
        """SD 1.x checkpoints load as StableDiffusionPipeline; SDXL checkpoints (Pony, Illustrious, ...) and files
        whose family cannot be detected (e.g. ``.ckpt``) keep loading as StableDiffusionXLPipeline."""
        if detect_single_file_architecture(model_path) == "sd15":
            return StableDiffusionPipeline
        return StableDiffusionXLPipeline

    def _load_single_file_pipeline(self, model_path: Path, torch_dtype: torch.dtype) -> Any:
        pipeline_class = self._single_file_pipeline_class(model_path)
        pipe = pipeline_class.from_single_file(
            str(model_path),
            use_safetensors=ILLUSTRIOUS_USE_SAFETENSORS,
            torch_dtype=torch_dtype,
        )
        self._configure_scheduler(pipe, model_path)
        return pipe

    def _configure_scheduler(self, pipe: Any, model_path: Path) -> None:
        """Replace the checkpoint's scheduler with the configured KSample sampler (v-prediction when the
        checkpoint file name says so)."""
        if (
            ILLUSTRIOUS_FORCE_V_PREDICTION_BY_NAME
            and self._looks_like_v_prediction_checkpoint(model_path)
        ):
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
                pipe.scheduler.config,
                prediction_type="v_prediction",
                rescale_betas_zero_snr=True,
                timestep_spacing="leading",
            )
        elif self.sampler_name == "euler_ancestral":
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
                pipe.scheduler.config,
                timestep_spacing="leading",
            )
        elif self.sampler_name == "euler":
            pipe.scheduler = EulerDiscreteScheduler.from_config(
                pipe.scheduler.config,
                timestep_spacing="leading",
            )

    def _place_pipeline_on_device(self) -> None:
        assert self.pipe is not None
        if (
            MODEL_CPU_OFFLOAD
            and hasattr(self.pipe, "enable_model_cpu_offload")
            and self.device.startswith("cuda")
        ):
            try:
                self.pipe.enable_model_cpu_offload()
                return
            except ImportError:
                pass
        self.pipe = self.pipe.to(self.device)

    def ensure_model(self, model: ModelSpec) -> None:
        if (
            self.current_model
            and self.current_model.id == model.id
            and self.pipe is not None
        ):
            return

        torch_dtype = self._effective_torch_dtype(model)
        model_path = Path(model.path)
        if model_path.is_file():
            self.pipe = self._load_single_file_pipeline(model_path, torch_dtype)
        else:
            self.pipe = AutoPipelineForText2Image.from_pretrained(
                str(model_path), torch_dtype=torch_dtype
            )
        self._place_pipeline_on_device()
        if hasattr(self.pipe, "enable_attention_slicing"):
            self.pipe.enable_attention_slicing()
        self.current_model = model
        self.loaded_adapters.clear()
        self.loaded_textual_inversions.clear()
        self.active_adapter_signature = None

    def unload(self) -> None:
        """Release the current pipeline, adapters, textual inversions, and CUDA cache."""
        pipe = self.pipe
        self.pipe = None
        self.current_model = None
        self.loaded_adapters.clear()
        self.loaded_textual_inversions.clear()
        self.active_adapter_signature = None
        self._cancel_event.clear()
        if pipe is not None:
            try:
                pipe.to("cpu")
            except Exception:
                pass
            del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def cancel_current_generation(self) -> None:
        self._cancel_event.set()

    def _summarize_adapter_file(self, path: Path) -> str:
        try:
            if path.suffix.casefold() == ".safetensors":
                try:
                    from safetensors import safe_open
                except Exception:
                    return "safetensors header unavailable"
                with safe_open(str(path), framework="pt", device="cpu") as handle:
                    keys = list(handle.keys())
                preview = ", ".join(keys[:8])
                lora_key_count = sum(1 for key in keys if "lora" in key.casefold())
                return (
                    f"keys={len(keys)}, lora_keys={lora_key_count}, preview=[{preview}]"
                )
            payload = torch.load(str(path), map_location="cpu", weights_only=False)
            if isinstance(payload, dict):
                keys = list(payload.keys())
                preview = ", ".join(str(key) for key in keys[:8])
                lora_key_count = sum(
                    1
                    for key in keys
                    if isinstance(key, str) and "lora" in key.casefold()
                )
                return (
                    f"keys={len(keys)}, lora_keys={lora_key_count}, preview=[{preview}]"
                )
            return f"payload_type={type(payload).__name__}"
        except Exception as exc:
            return f"summary_failed={exc}"


    def _read_adapter_state_dict(self, path: Path) -> dict[str, torch.Tensor]:
        if path.suffix.casefold() == ".safetensors":
            from safetensors.torch import load_file

            return load_file(str(path), device="cpu")
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError(f"LoRA file payload must be a state dict, got {type(payload).__name__}")
        return payload

    def _contiguous_adapter_state_dict(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Return tensors in a layout accepted by safetensors.

        Some LoCon folding operations above intentionally use transpose/reshape
        views. Those views are mathematically correct, but safetensors requires
        tensors to be packed contiguously before serialization.
        """
        contiguous_state_dict: dict[str, torch.Tensor] = {}
        for key, tensor in state_dict.items():
            if isinstance(tensor, torch.Tensor):
                tensor = tensor.detach()
                if not tensor.is_contiguous():
                    tensor = tensor.contiguous()
            contiguous_state_dict[key] = tensor
        return contiguous_state_dict

    def _write_adapter_state_dict(self, state_dict: dict[str, torch.Tensor], path: Path) -> None:
        if path.suffix.casefold() == ".safetensors":
            from safetensors.torch import save_file

            save_file(self._contiguous_adapter_state_dict(state_dict), str(path))
        else:
            torch.save(state_dict, str(path))

    def _fold_lora_mid_weights_for_diffusers(
        self, state_dict: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], int]:
        """Fold Kohya/ComfyUI LoCon ``lora_mid`` tensors into ``lora_down`` tensors.

        Diffusers' LoRA loader expects a two-factor adapter (down/up). Some
        ComfyUI-compatible LoCon files store convolutional adapters as three
        factors (down/mid/up). For these files, leaving ``lora_mid.weight`` in
        the state dict makes Diffusers fail during key conversion with
        "keys have not been correctly renamed".

        The transformation below is mathematically equivalent to ComfyUI's
        LoCon merge path: it materializes the effective down convolution and
        removes the separate mid tensor before handing the file to Diffusers.
        """
        converted = dict(state_dict)
        folded = 0
        mid_suffix = ".lora_mid.weight"

        for mid_key in [key for key in converted if key.endswith(mid_suffix)]:
            prefix = mid_key[: -len(mid_suffix)]
            down_key = f"{prefix}.lora_down.weight"
            if down_key not in converted:
                continue

            down = converted[down_key].detach().cpu()
            mid = converted[mid_key].detach().cpu()

            if mid.ndim == 4:
                if down.ndim == 2:
                    down_2d = down
                    input_dim = down.shape[1]
                elif down.ndim == 4:
                    if any(size != 1 for size in down.shape[2:]):
                        raise ValueError(
                            f"Cannot fold {mid_key}: lora_down already has a non-1x1 kernel, "
                            f"shape={tuple(down.shape)}"
                        )
                    down_2d = down.reshape(down.shape[0], down.shape[1])
                    input_dim = down.shape[1]
                else:
                    raise ValueError(
                        f"Unsupported lora_down rank for {down_key}: shape={tuple(down.shape)}"
                    )

                if down_2d.shape[0] != mid.shape[1]:
                    raise ValueError(
                        f"Cannot fold {mid_key}: down rank {down_2d.shape[0]} does not match "
                        f"mid input rank {mid.shape[1]}"
                    )
                effective_down = (
                    down_2d.transpose(0, 1)
                    @ mid.transpose(0, 1).flatten(start_dim=1)
                ).reshape(input_dim, mid.shape[0], mid.shape[2], mid.shape[3]).transpose(0, 1)
            elif mid.ndim == 2:
                if down.ndim != 2:
                    raise ValueError(
                        f"Unsupported linear lora_down shape for {down_key}: shape={tuple(down.shape)}"
                    )
                if down.shape[0] != mid.shape[1]:
                    raise ValueError(
                        f"Cannot fold {mid_key}: down rank {down.shape[0]} does not match "
                        f"mid input rank {mid.shape[1]}"
                    )
                effective_down = mid @ down
            else:
                raise ValueError(f"Unsupported lora_mid shape for {mid_key}: shape={tuple(mid.shape)}")

            converted[down_key] = effective_down.to(dtype=down.dtype)
            del converted[mid_key]
            folded += 1

        return converted, folded

    @contextmanager
    def _diffusers_adapter_load_file(self, path: Path):
        if not path.is_file() or path.suffix.casefold() not in {".safetensors", ".pt", ".pth", ".bin"}:
            yield path
            return

        state_dict = self._read_adapter_state_dict(path)
        if not any(key.endswith(".lora_mid.weight") for key in state_dict):
            yield path
            return

        converted, folded = self._fold_lora_mid_weights_for_diffusers(state_dict)
        if folded == 0:
            yield path
            return

        with tempfile.TemporaryDirectory(prefix="diffusion-lora-locon-") as tmpdir:
            load_path = Path(tmpdir) / path.name
            self._write_adapter_state_dict(converted, load_path)
            logger.info(
                "Folded %d lora_mid tensors into lora_down tensors before Diffusers LoRA load: %s",
                folded,
                path,
            )
            yield load_path

    def _load_single_adapter(self, spec: LoRASpec) -> None:
        assert self.pipe is not None
        path = Path(spec.path)
        try:
            if path.is_file():
                with self._diffusers_adapter_load_file(path) as load_path:
                    self.pipe.load_lora_weights(
                        str(load_path.parent), weight_name=load_path.name, adapter_name=spec.id
                    )
            else:
                self.pipe.load_lora_weights(str(path), adapter_name=spec.id)
        except Exception as exc:
            summary = (
                self._summarize_adapter_file(path)
                if path.is_file()
                else "directory-based adapter"
            )
            guidance = ""
            lowered_summary = summary.casefold()
            if (
                "preview=[clip_g, clip_l]" in lowered_summary
                or "preview=[clip_l, clip_g]" in lowered_summary
            ):
                guidance = (
                    " This file looks like a prompt embedding/textual inversion (clip_l/clip_g only), "
                    "not a diffusers-loadable LoRA adapter. In ComfyUI this kind of asset is typically used "
                    "through embedding/text encoding, not as a standard diffusers LoRA load."
                )
            logger.exception(
                "Failed to load LoRA adapter '%s' from '%s'. %s", spec.id, path, summary
            )
            raise AdapterLoadError(
                f"Failed to load LoRA '{spec.name}' from '{path}': {exc}. {summary}.{guidance}"
            ) from exc
        self.loaded_adapters[spec.id] = LoadedAdapter(spec=spec, device=self.device)

    def _encoder_embedding_width(self, text_encoder: Any) -> int:
        return int(text_encoder.get_input_embeddings().weight.shape[1])

    def _resize_token_embeddings(self, text_encoder: Any, size: int) -> None:
        try:
            text_encoder.resize_token_embeddings(size, mean_resizing=False)
        except TypeError:
            text_encoder.resize_token_embeddings(size)

    def _encoder_input_device(self, text_encoder: Any) -> torch.device:
        if self.device.startswith("cuda"):
            return torch.device(self.device)
        embedding_layer = text_encoder.get_input_embeddings()
        device = getattr(getattr(embedding_layer, "weight", None), "device", None)
        if isinstance(device, torch.device) and device.type != "meta":
            return device
        if isinstance(device, str) and device != "meta":
            return torch.device(device)
        return torch.device(self.device)

    def _register_textual_inversion_tokens(
        self,
        tokenizer: Any,
        text_encoder: Any,
        base_token: str,
        embedding_vectors: torch.Tensor,
    ) -> list[str]:
        token_names = [
            base_token,
            *[
                f"{base_token}_{index}"
                for index in range(1, embedding_vectors.shape[0])
            ],
        ]
        tokenizer.add_tokens(token_names)
        self._resize_token_embeddings(text_encoder, len(tokenizer))

        embedding_layer = text_encoder.get_input_embeddings()
        token_ids = tokenizer.convert_tokens_to_ids(token_names)
        target_device = embedding_layer.weight.device
        target_dtype = embedding_layer.weight.dtype

        with torch.no_grad():
            for row_index, token_id in enumerate(token_ids):
                embedding_layer.weight[token_id] = embedding_vectors[row_index].to(
                    device=target_device,
                    dtype=target_dtype,
                )

        return token_names

    def _load_single_textual_inversion(
        self, reference: PromptEmbeddingReference
    ) -> None:
        assert self.pipe is not None
        # SDXL: two tokenizer/text encoder pairs (CLIP-L + OpenCLIP bigG); SD 1.x: one CLIP-L pair.
        dual_encoder = self._supports_long_prompt_embeds()
        if not dual_encoder and not all(
            hasattr(self.pipe, attribute) for attribute in ("tokenizer", "text_encoder")
        ):
            raise PromptEmbeddingResolutionError(
                "Prompt embeddings require a pipeline with a tokenizer/text encoder pair."
            )

        state = load_prompt_embedding_file(reference.path)
        vectors = resolve_prompt_embedding_vectors(
            state,
            clip_l_dim=self._encoder_embedding_width(self.pipe.text_encoder),
            clip_g_dim=(
                self._encoder_embedding_width(self.pipe.text_encoder_2)
                if dual_encoder
                else None
            ),
        )

        token_names = self._register_textual_inversion_tokens(
            self.pipe.tokenizer,
            self.pipe.text_encoder,
            reference.token,
            vectors.clip_l,
        )
        if dual_encoder:
            token_names_b = self._register_textual_inversion_tokens(
                self.pipe.tokenizer_2,
                self.pipe.text_encoder_2,
                reference.token,
                vectors.clip_g,
            )
            if len(token_names_b) > len(token_names):
                token_names = token_names_b

        resolved = str(reference.path.resolve()).casefold()
        self.loaded_textual_inversions[resolved] = LoadedTextualInversion(
            token=reference.token,
            path=str(reference.path),
            tokens=token_names,
        )

    def ensure_textual_inversions(
        self,
        requested: list[PromptEmbeddingReference],
    ) -> list[str]:
        assert self.pipe is not None
        resolved_tokens: list[str] = []
        for reference in requested:
            resolved = str(reference.path.resolve()).casefold()
            loaded = self.loaded_textual_inversions.get(resolved)
            if loaded is None:
                self._load_single_textual_inversion(reference)
                loaded = self.loaded_textual_inversions[resolved]
            resolved_tokens.append(loaded.token)
        return resolved_tokens

    def expand_prompt_embeddings(self, text: str) -> str:
        expanded = text
        for loaded in sorted(
            self.loaded_textual_inversions.values(),
            key=lambda item: len(item.token),
            reverse=True,
        ):
            replacement = " ".join(loaded.tokens)
            expanded = re.sub(
                rf"(?<![A-Za-z0-9_]){re.escape(loaded.token)}(?![A-Za-z0-9_])",
                replacement,
                expanded,
            )
        return expanded

    def ensure_adapters(
        self,
        requested: list[LoRASpec],
        adapter_weights: list[float] | None = None,
    ) -> list[str]:
        assert self.pipe is not None
        requested_ids = [spec.id for spec in requested]
        resolved_weights = adapter_weights or [LORA_WEIGHT] * len(requested)
        if len(resolved_weights) != len(requested):
            raise ValueError("adapter_weights length must match requested adapters.")

        if not requested_ids:
            if hasattr(self.pipe, "disable_lora"):
                self.pipe.disable_lora()
            self.active_adapter_signature = ((), ())
            return []

        for spec in requested:
            loaded = self.loaded_adapters.get(spec.id)
            if loaded is None:
                self._load_single_adapter(spec)
            elif loaded.device != self.device and hasattr(self.pipe, "set_lora_device"):
                self.pipe.set_lora_device(adapter_names=[spec.id], device=self.device)
                loaded.device = self.device

        inactive_ids = [
            adapter_id
            for adapter_id in self.loaded_adapters.keys()
            if adapter_id not in requested_ids
        ]
        if (
            inactive_ids
            and hasattr(self.pipe, "set_lora_device")
            and self.device.startswith("cuda")
        ):
            self.pipe.set_lora_device(adapter_names=inactive_ids, device="cpu")
            for adapter_id in inactive_ids:
                self.loaded_adapters[adapter_id].device = "cpu"

        requested_signature = (
            tuple(requested_ids),
            tuple(float(weight) for weight in resolved_weights),
        )
        if self.active_adapter_signature == requested_signature:
            if hasattr(self.pipe, "enable_lora"):
                self.pipe.enable_lora()
            return requested_ids

        with torch.no_grad():
            self.pipe.set_adapters(requested_ids, adapter_weights=resolved_weights)
        if hasattr(self.pipe, "enable_lora"):
            self.pipe.enable_lora()
        self.active_adapter_signature = requested_signature
        return requested_ids

    def _supports_long_prompt_embeds(self) -> bool:
        assert self.pipe is not None
        required = ("tokenizer", "tokenizer_2", "text_encoder", "text_encoder_2")
        return all(hasattr(self.pipe, attribute) for attribute in required)

    def _module_device(self, module: Any) -> torch.device:
        try:
            parameter = next(module.parameters())
            return parameter.device
        except Exception:
            return torch.device(self.device)

    def _encode_long_text(
        self, prompt: str, tokenizer: Any, text_encoder: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_length = int(getattr(tokenizer, "model_max_length", 77))
        if max_length < 3:
            raise ValueError(
                "Tokenizer model_max_length is too small for long-prompt encoding."
            )

        bos_id = getattr(tokenizer, "bos_token_id", None)
        if bos_id is None:
            bos_id = getattr(tokenizer, "cls_token_id", None)
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if eos_id is None:
            eos_id = getattr(tokenizer, "sep_token_id", None)
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = eos_id if eos_id is not None else 0
        if bos_id is None:
            bos_id = eos_id if eos_id is not None else pad_id
        if eos_id is None:
            eos_id = pad_id

        tokenized = tokenizer(
            prompt or "",
            add_special_tokens=False,
            truncation=False,
            verbose=False,
        )
        raw_ids = tokenized.input_ids
        if raw_ids and isinstance(raw_ids[0], list):
            raw_ids = raw_ids[0]

        content_limit = max_length - 2
        chunks = [
            raw_ids[index : index + content_limit]
            for index in range(0, len(raw_ids), content_limit)
        ] or [[]]
        hidden_states: list[torch.Tensor] = []
        pooled_output: torch.Tensor | None = None
        module_device = self._encoder_input_device(text_encoder)

        for chunk in chunks:
            chunk_ids = [bos_id, *chunk[:content_limit], eos_id]
            attention_mask = [1] * len(chunk_ids)
            while len(chunk_ids) < max_length:
                chunk_ids.append(pad_id)
                attention_mask.append(0)

            input_ids = torch.tensor(
                [chunk_ids], device=module_device, dtype=torch.long
            )
            mask_tensor = torch.tensor(
                [attention_mask], device=module_device, dtype=torch.long
            )

            with torch.no_grad():
                encoded = text_encoder(
                    input_ids,
                    attention_mask=mask_tensor,
                    output_hidden_states=True,
                    return_dict=True,
                )
            hidden_states.append(encoded.hidden_states[-2])

            pooled_candidate = getattr(encoded, "text_embeds", None)
            if pooled_candidate is None:
                pooled_candidate = getattr(encoded, "pooler_output", None)
            if pooled_candidate is None:
                first_output = encoded[0]
                pooled_candidate = (
                    first_output[:, 0]
                    if getattr(first_output, "ndim", 0) == 3
                    else first_output
                )
            pooled_output = pooled_candidate

        assert pooled_output is not None
        return torch.cat(hidden_states, dim=1), pooled_output

    def _pad_prompt_embeds(
        self, prompt_embeds: torch.Tensor, target_length: int
    ) -> torch.Tensor:
        if prompt_embeds.shape[1] >= target_length:
            return prompt_embeds
        pad = prompt_embeds[:, -1:, :].repeat(
            1, target_length - prompt_embeds.shape[1], 1
        )
        return torch.cat([prompt_embeds, pad], dim=1)

    def _build_long_prompt_kwargs(
        self, prompt: str, negative_prompt: str
    ) -> dict[str, Any]:
        assert self.pipe is not None
        prompt_embed_a, _ = self._encode_long_text(
            prompt, self.pipe.tokenizer, self.pipe.text_encoder
        )
        prompt_embed_b, pooled_prompt = self._encode_long_text(
            prompt, self.pipe.tokenizer_2, self.pipe.text_encoder_2
        )
        negative_embed_a, _ = self._encode_long_text(
            negative_prompt or "", self.pipe.tokenizer, self.pipe.text_encoder
        )
        negative_embed_b, negative_pooled_prompt = self._encode_long_text(
            negative_prompt or "",
            self.pipe.tokenizer_2,
            self.pipe.text_encoder_2,
        )

        target_sequence_length = max(
            prompt_embed_a.shape[1],
            prompt_embed_b.shape[1],
            negative_embed_a.shape[1],
            negative_embed_b.shape[1],
        )

        prompt_embed_a = self._pad_prompt_embeds(prompt_embed_a, target_sequence_length)
        prompt_embed_b = self._pad_prompt_embeds(prompt_embed_b, target_sequence_length)
        negative_embed_a = self._pad_prompt_embeds(
            negative_embed_a, target_sequence_length
        )
        negative_embed_b = self._pad_prompt_embeds(
            negative_embed_b, target_sequence_length
        )

        prompt_embeds = torch.cat([prompt_embed_a, prompt_embed_b], dim=-1)
        negative_prompt_embeds = torch.cat([negative_embed_a, negative_embed_b], dim=-1)

        target_dtype = getattr(
            getattr(self.pipe, "unet", None), "dtype", prompt_embeds.dtype
        )
        prompt_embeds = prompt_embeds.to(device=self.device, dtype=target_dtype)
        negative_prompt_embeds = negative_prompt_embeds.to(
            device=self.device, dtype=target_dtype
        )
        pooled_prompt = pooled_prompt.to(device=self.device, dtype=target_dtype)
        negative_pooled_prompt = negative_pooled_prompt.to(
            device=self.device, dtype=target_dtype
        )

        return {
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt,
            "negative_pooled_prompt_embeds": negative_pooled_prompt,
        }

    def generate(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        steps: int,
        guidance_scale: float,
        seed: int | None,
        progress_callback: Callable[[int, int], None] | None = None,
        step_callback: StepCallback | None = None,
    ) -> tuple[Image.Image, int]:
        assert self.pipe is not None

        prompt = self.expand_prompt_embeddings(prompt)
        negative_prompt = self.expand_prompt_embeddings(negative_prompt)

        self._cancel_event.clear()

        actual_seed = (
            seed if seed is not None else random.SystemRandom().randint(1, 2**31 - 1)
        )
        generator_device = self.device if self._device_type() == "cuda" else "cpu"
        generator = torch.Generator(device=generator_device)
        generator = generator.manual_seed(actual_seed)

        call_kwargs = {
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
        }
        if self._supports_long_prompt_embeds():
            call_kwargs.update(self._build_long_prompt_kwargs(prompt, negative_prompt))
        else:
            call_kwargs.update(
                {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt or None,
                }
            )

        last_notified_step: int | None = None

        def notify_progress(
            step: int, total_steps: int, timestep: Any | None = None
        ) -> None:
            nonlocal last_notified_step
            if last_notified_step == step:
                return
            last_notified_step = step
            if progress_callback is not None:
                progress_callback(step, total_steps)
            if step_callback is not None:
                raw_timestep: int | float | None
                if timestep is None:
                    raw_timestep = None
                elif hasattr(timestep, "item"):
                    raw_timestep = timestep.item()
                else:
                    raw_timestep = timestep
                step_callback(
                    StepProgress(
                        step=step, total_steps=total_steps, timestep=raw_timestep
                    )
                )

        def check_cancelled() -> None:
            if self._cancel_event.is_set():
                raise GenerationCancelled("Generation cancelled.")

        notify_progress(0, steps)
        check_cancelled()

        def on_step_end(
            _: Any, step_index: int, timestep: Any, callback_kwargs: dict[str, Any]
        ) -> dict[str, Any]:
            notify_progress(step_index + 1, steps, timestep=timestep)
            check_cancelled()
            return callback_kwargs

        def legacy_callback(step_index: int, timestep: Any, _: Any) -> None:
            notify_progress(step_index + 1, steps, timestep=timestep)
            check_cancelled()

        parameters = inspect.signature(self.pipe.__call__).parameters
        if "callback_on_step_end" in parameters:
            result = self.pipe(
                **call_kwargs,
                callback_on_step_end=on_step_end,
            )
        elif "callback" in parameters:
            result = self.pipe(
                **call_kwargs, callback=legacy_callback, callback_steps=1
            )
        else:
            result = self.pipe(**call_kwargs)

        check_cancelled()
        notify_progress(steps, steps)

        return result.images[0], actual_seed

    @staticmethod
    def image_to_base64(image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
