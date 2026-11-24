# diffusion-lora-runtime

Standalone Python package for Diffusers model loading, LoRA loading, prompt asset resolution, and KSample-style image generation.

This package keeps the part you asked for: diffusion model loading, LoRA loading, A1111-style prompt LoRA resolution, textual-inversion prompt embedding resolution, KSample-style generation config, and in-memory model reuse. FastAPI, websocket transport, desktop storage, output browsing, and frontend prompt-category logic were deliberately left out.

## What was extracted

Copied/adapted from the source backend as directly as possible:

- `diffusion_runtime.py` — Diffusers runtime, SD 1.x / SDXL single-file model loading, LoRA adapters, textual inversion registration, long prompt embedding generation, sampler callback handling, image generation.
- `prompt_lora.py` — resolves `<lora:name:weight>` tags, searches LoRA files by filename stem, reads optional `.txt` sidecar text, and returns LoRA weights.
- `prompt_embedding.py` — resolves `embedding:name` tags, loads `.safetensors`/`.pt` prompt embeddings, supports SDXL `clip_l`/`clip_g` tensors and single-tensor SD 1.x embeddings.
- `prompt_assets.py` and `normalization.py` — strict recursive asset lookup and prompt cleanup helpers.

Added for package use:

- `generator.py` — a small public API with `generate()` and `unload()`.
- `schemas.py` — `ModelSpec`, `LoRASpec`, `IPAdapterSpec`, `KSampleConfig`, `GenerateResult`, and path helpers.
- `checkpoint_inspection.py` — reads a `.safetensors` header (pure Python, no torch) to tell SD 1.x from SDXL single-file checkpoints, and owns the rule "which pipeline family does the runtime load this model as" (`pipeline_architecture`).
- `ip_adapter.py` — IP-Adapter pieces that need no pipeline: the CLIP ViT-H/14 image encoder configuration, image encoder loading (ComfyUI single file or transformers folder) and reference image opening. Loading the adapter into a pipeline is `DiffusionRuntime.ensure_ip_adapter`.

The package's public names are resolved lazily (PEP 562): `import diffusion_lora_runtime` and its torch-free modules (`prompt_lora`, `normalization`, `prompt_assets`, `schemas`, `checkpoint_inspection`, `ip_adapter`) never import torch or diffusers; they load the first time a name that needs them (`DiffusionGenerator`, `AdapterLoadError`, `GenerationCancelled`, `PromptEmbeddingResolutionError`) is accessed.

## Changes in 0.4.0

- IP-Adapter support: `generate(..., ip_adapter=IPAdapterSpec(...))` guides the image with reference images (SD 1.x and SDXL), see "IP-Adapter (reference images)".
- LoRA changes no longer reload the model: only the added LoRAs are loaded, only the removed ones are deleted, and a weight-only change just re-applies the weights (see "Memory behavior").
- `pipeline_architecture(path)` / `single_file_pipeline_architecture(path)`: the family (`"sd15"` / `"sdxl"`) the runtime loads a model as, torch-free, also used by the runtime's own loader.

## Installation

From this folder:

```bash
pip install -e .
```

For CUDA, install the `torch` build that matches your machine first, then install this package. The runtime now fails fast if it would otherwise fall back to CPU by accident, because SDXL with FP16 on CPU usually appears to hang.

Quick CUDA check inside the environment that will run generation:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

If that prints `False`, fix the PyTorch install/index for the consuming project before running generation.

## Basic usage

```python
from diffusion_lora_runtime import DiffusionGenerator, KSampleConfig, LoRASpec

engine = DiffusionGenerator()  # auto-selects CUDA when available; CPU fallback is disabled by default

result = engine.generate(
    model_path=r"D:\Anything\models\illustrious-vpred.safetensors",
    prompt="masterpiece, best quality, <lora:my_style:0.8>",
    negative_prompt="low quality, blurry",
    loras=[
        LoRASpec.from_path(r"D:\Anything\ComfyUI\ComfyUI\models\loras\detail.safetensors", weight=0.65),
    ],
    prompt_lora_dir=r"D:\Anything\ComfyUI\ComfyUI\models\loras",
    prompt_embedding_dir=r"D:\Anything\ComfyUI\ComfyUI\models\embeddings",
    ksample=KSampleConfig(width=1024, height=1024, steps=30, cfg=7.0, seed=123456),
    output_path="output.png",
)

print(result.seed)
print(result.model_path)
print([(lora.name, lora.path, lora.weight) for lora in result.loras])
print(result.ksample)

engine.unload()
```

## Public API

### `DiffusionGenerator.generate(...)`

Main method. It loads or reuses the current model and LoRA set, resolves prompt assets, samples one image, and returns a `GenerateResult`.

Important arguments:

- `model_path`: Diffusers model directory or single-file checkpoint path.
- `prompt`: final positive prompt string.
- `negative_prompt`: negative prompt string.
- `loras`: explicit LoRAs. Each item can be a string/path or `LoRASpec`.
- `prompt_lora_dir`: folder used to resolve inline `<lora:name:weight>` tags.
- `prompt_embedding_dir`: folder used to resolve inline `embedding:name` tags.
- `ksample`: `KSampleConfig(width, height, steps, cfg, seed, sampler_name, scheduler)`.
- `output_path`: optional image save path. If omitted, the image is returned but not saved.
- `progress_callback`: optional callable receiving `(current_step, total_steps)`.
- `step_callback`: optional callable receiving a `StepProgress(step, total_steps, timestep)` object after each scheduler step.
- `ip_adapter`: optional `IPAdapterSpec`: reference images applied through an IP-Adapter (see below). `None` generates without one (an IP-Adapter loaded by an earlier call is unloaded; the base model stays loaded).

Shortcut overrides are also accepted: `width`, `height`, `steps`, `cfg`, and `seed`. They override the values inside `ksample` for that call.

### `DiffusionGenerator.unload()`

Cancels any current generation flag, drops the Diffusers pipeline, clears LoRA / IP-Adapter / textual-inversion bookkeeping, runs `gc.collect()`, and empties the CUDA cache when CUDA is available.

### Checkpoint inspection

```python
from diffusion_lora_runtime import (
    detect_single_file_architecture,
    pipeline_architecture,
    read_safetensors_keys,
    single_file_pipeline_architecture,
)

detect_single_file_architecture("realisticVision.safetensors")  # "sd15"
detect_single_file_architecture("illustriousXL.safetensors")    # "sdxl"
detect_single_file_architecture("model.ckpt")                   # None (never parsed)
read_safetensors_keys("model.safetensors")                       # tensor names, header order

single_file_pipeline_architecture("model.ckpt")                 # "sdxl": undetected files load as SDXL
pipeline_architecture("realisticVision.safetensors")             # "sd15"
pipeline_architecture("diffusers-folder/")                       # from model_index.json "_class_name"
pipeline_architecture("missing.safetensors")                     # None
```

They only read the safetensors header (8-byte little-endian length + JSON) or a Diffusers folder's `model_index.json`; no tensor is loaded and torch is not imported. `read_safetensors_keys` raises `SafetensorsHeaderError` for malformed headers; `detect_single_file_architecture` returns `None` for anything it cannot classify (see "Single-file checkpoints" below).

`pipeline_architecture(path)` is the single owner of the rule "which pipeline family does the runtime load this model as": an existing file is `single_file_pipeline_architecture` (`"sd15"` for SD 1.x files, `"sdxl"` for everything else, exactly the pipeline class the loader picks); a Diffusers folder is `"sdxl"` for `StableDiffusionXL*` and `"sd15"` for `StableDiffusionPipeline` in its `model_index.json`, else `None`; a missing path is `None`. Consumers use it before any model is loaded, e.g. to pick an IP-Adapter made for that family.

### IP-Adapter (reference images)

```python
from diffusion_lora_runtime import IPAdapterSpec

result = engine.generate(
    model_path=r"D:\Anything\models\illustriousXL.safetensors",
    prompt="1girl, portrait",
    ip_adapter=IPAdapterSpec(
        adapter_path=r"D:\Anything\ComfyUI\ComfyUI\models\ipadapter\ip-adapter-plus-face_sdxl_vit-h.safetensors",
        image_encoder_path=r"D:\Anything\ComfyUI\ComfyUI\models\clip_vision\CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors",
        image_paths=("face_front.png", "face_side.png"),
        scale=0.6,
        architecture="sdxl",
    ),
)
```

`IPAdapterSpec` fields:

- `adapter_path`: IP-Adapter weights in the h94/IP-Adapter layout (`.safetensors` with `image_proj.*` / `ip_adapter.*` keys, or `.bin`), e.g. `ip-adapter-plus-face_sdxl_vit-h.safetensors`, `ip-adapter-plus_sd15.safetensors`. They are loaded with Diffusers `load_ip_adapter(<folder>, subfolder="", weight_name=<file name>, image_encoder_folder=None)`.
- `image_encoder_path`: the CLIP vision encoder the adapter was trained with (OpenCLIP ViT-H/14 for every `vit-h` / SD 1.5 h94 adapter), either
  - a single `.safetensors` file in the transformers key layout (`vision_model.*`, `visual_projection.weight`), i.e. a ComfyUI `models/clip_vision` file such as `CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors`: it carries no configuration, so it is built with `VIT_H_14_IMAGE_ENCODER_CONFIG` (the h94 `models/image_encoder/config.json`) and loaded strictly (a stored `vision_model.embeddings.position_ids` buffer is tolerated); the model is created without allocating weights and takes the file's tensors, so the 2.5 GB file is held in memory once;
  - or a transformers model folder with its `config.json` (e.g. h94 `models/image_encoder`), loaded with `CLIPVisionModelWithProjection.from_pretrained`.
  A file of another layout (e.g. OpenCLIP `visual.*` keys) raises `ImageEncoderLayoutError`.
- `image_paths`: one or more reference images, in order. They are opened as RGB before anything is loaded; a missing or unreadable file (or none) raises `ReferenceImageError` (a `ValueError`). All of them go to the pipeline as one adapter's images (`ip_adapter_image=[[image, ...]]`).
- `scale`: adapter strength (Diffusers `set_ip_adapter_scale`), default `0.6`.
- `architecture`: `"sd15"` / `"sdxl"`, the family the adapter was made for, checked against the loaded pipeline (two text encoders: SDXL, one: SD 1.x). A mismatch raises `AdapterLoadError` ("IP-Adapter made for SDXL cannot be used with an SD 1.5 model."). `None` skips the check.

Loading registers the image encoder (created with the pipeline dtype) on the pipeline before `load_ip_adapter`, so Diffusers does not look for its own encoder and creates the `CLIPImageProcessor` itself. Under model CPU offload the encoder is created on the CPU and `enable_model_cpu_offload()` is called again so it joins the offload hooks; otherwise it is placed on the device. Every IP-Adapter failure (missing weights, bad encoder, Diffusers errors) is raised as `AdapterLoadError`, after restoring the pipeline (`unload_ip_adapter`).


## Progress callbacks

Use `progress_callback` for a simple scalar counter:

```python
result = engine.generate(
    model_path="model.safetensors",
    prompt="cat",
    ksample=KSampleConfig(steps=30),
    progress_callback=lambda step, total: print(f"{step}/{total}"),
)
```

Use `step_callback` when the caller wants a richer event object:

```python
from diffusion_lora_runtime import StepProgress

def on_step(event: StepProgress) -> None:
    print(event.step, event.total_steps, event.fraction, event.timestep)

result = engine.generate(
    model_path="model.safetensors",
    prompt="cat",
    ksample=KSampleConfig(steps=30),
    step_callback=on_step,
)
```

The callback is invoked at step `0` before sampling and then at each Diffusers scheduler step. Cancellation is also checked from this step callback path, so callers can expose a cancel button without waiting for the whole generation to finish.

## Device and dtype behavior

By default:

```python
engine = DiffusionGenerator()
```

means:

- prefer CUDA if `torch.cuda.is_available()` is true;
- otherwise prefer MPS on Apple Silicon if available;
- otherwise select CPU but raise a clear error before model loading.

This prevents the common accidental state where a CPU-only PyTorch wheel is installed, Diffusers loads an FP16 pipeline, and the process appears stuck at `0%`.

For explicit CPU debugging only:

```python
engine = DiffusionGenerator(device="cpu", allow_cpu=True)
result = engine.generate(
    model_path="model.safetensors",
    prompt="cat",
    torch_dtype="float32",
)
```

If `allow_cpu=True` and the caller still asks for `float16` or `bfloat16`, the runtime promotes the model load to `float32`.

## Memory behavior

`generate()` keeps the loaded model, LoRAs, IP-Adapter and textual inversions between calls, and only changes what a call changes:

- Base model: only a different model path, dtype, sampler or scheduler unloads the previous pipeline (and everything loaded into it) before loading the new one.
- LoRAs (explicit and inline prompt LoRAs, merged first) are updated in place, never by reloading the model:
  - an unchanged set (same LoRAs, order and weights) calls nothing;
  - only added LoRAs are loaded; only removed ones are deleted from the pipeline (`delete_adapters`, so memory stays bounded: nothing is parked on the CPU); a LoRA whose id now names another file is replaced;
  - a weight-only change calls `set_adapters(ids, weights)` and nothing else;
  - an empty set deletes the loaded LoRAs and calls `disable_lora()`; the next set enables LoRA layers again.
- IP-Adapter: loaded, swapped (the previous one is unloaded first; the image encoder module is reused when its path is unchanged), re-scaled (`set_ip_adapter_scale` only) or unloaded (`ip_adapter=None`), never by reloading the model. Unloading also releases the cached image encoder.
- Textual inversions stay loaded across LoRA and IP-Adapter changes.
- Explicit LoRAs and inline prompt LoRAs are combined by resolved path. Inline tags can override a duplicate explicit LoRA's weight because this matches the original backend flow.
- A failed LoRA load is cleaned up (`delete_adapters` of what was injected) and retried by the next request; a failed delete is reported as `AdapterLoadError` and retried too.
- `unload()` always releases the current runtime state.

## Prompt resolution

### Inline LoRA tags

Supported inline syntax:

```text
<lora:name:weight>
<lora:name>
```

Rules:

- Requires `prompt_lora_dir` if the prompt contains inline LoRA tags.
- Recursively searches files under `prompt_lora_dir` by strict filename stem match.
- Ignores `.txt` sidecar files during LoRA asset lookup.
- If `name.txt` exists beside the LoRA file, that text replaces the inline tag in the prompt.
- The LoRA file is added to the resolved LoRA list with its parsed weight.

The syntax itself is also available without any file lookup and without torch, for reading prompts written by other tools:

```python
from diffusion_lora_runtime.prompt_lora import extract_prompt_lora_specs, strip_prompt_lora_tags

extract_prompt_lora_specs("castle, <lora:add_detail:0.6>, <lora:kim_v_1>")  # [("add_detail", 0.6), ("kim_v_1", 1.0)]
strip_prompt_lora_tags("castle, <lora:add_detail:0.6>, sky")                 # "castle, sky"
```

### Prompt embeddings / textual inversion

Supported inline syntax:

```text
embedding:name
```

Rules:

- Requires `prompt_embedding_dir` if positive or negative prompts contain embedding tags.
- Recursively searches files under `prompt_embedding_dir` by strict filename stem match.
- Replaces the prompt text with generated `dlr_embed_*` tokens.
- Loads SDXL `clip_l`/`clip_g` embedding tensors when available.
- SDXL pipelines: registers textual inversion tokens into both tokenizers/text encoders (a single-tensor 768-wide SD 1.x embedding fills `clip_l`, `clip_g` stays zero).
- SD 1.x pipelines (one CLIP-L tokenizer/text encoder): registers the `clip_l` vectors only (A1111 `string_to_param`, a single tensor, or the `clip_l` part of an SDXL embedding); an embedding with only `clip_g` vectors is refused with `PromptEmbeddingResolutionError`.

## KSample config mapping

`KSampleConfig` stores the generation settings that matter for this extracted backend:

```python
KSampleConfig(
    width=1024,
    height=1024,
    steps=30,
    cfg=7.0,
    seed=None,
    sampler_name="euler_ancestral",
    scheduler="simple",
)
```

Mapping to Diffusers:

- `width` -> pipeline `width`
- `height` -> pipeline `height`
- `steps` -> pipeline `num_inference_steps`
- `cfg` -> pipeline `guidance_scale`
- `seed` -> `torch.Generator.manual_seed(...)`
- `sampler_name="euler_ancestral"` -> `EulerAncestralDiscreteScheduler`
- `sampler_name="euler"` -> `EulerDiscreteScheduler`
- `scheduler="simple"` is retained as metadata/source compatibility; the original backend also kept this as a constant.

## Notes

- Real image generation requires local model files and an environment with Torch/Diffusers installed.
- Single-file checkpoints: the family is detected from the `.safetensors` header (`detect_single_file_architecture`):
  - SD 1.x (`cond_stage_model.transformer.*` tensors, e.g. SD 1.5, SD 1.5 LCM/Hyper) loads through `StableDiffusionPipeline.from_single_file(...)`;
  - SDXL (`conditioner.embedders.1.*` or `model.diffusion_model.label_emb.*` tensors: SDXL 1.0, Pony, Illustrious, NoobAI, ...) loads through `StableDiffusionXLPipeline.from_single_file(...)`;
  - anything undetected (`.ckpt` files, SD 2.x, unknown layouts) keeps the original behavior: `StableDiffusionXLPipeline.from_single_file(...)`.
  Both single-file loaders then apply the same scheduler configuration (`euler_ancestral` / `euler`, `timestep_spacing="leading"`, v-prediction by file name below). Diffusers model directories load with `AutoPipelineForText2Image.from_pretrained(...)` and keep their scheduler.
- SD 1.x pipelines take the prompt as plain text (Diffusers truncates it at 77 tokens); the long-prompt chunked encoding is SDXL-only.
- If a checkpoint filename contains `vpred`, `v-pred`, `v_prediction`, or `v-prediction`, the runtime forces `prediction_type="v_prediction"` and `rescale_betas_zero_snr=True`, copied from the original Illustrious-specific behavior.
- Some ComfyUI LoRA/textual-inversion files may not be loadable by Diffusers. The original adapter error summarization was kept to make those failures easier to debug.

## Validation performed in this environment

- Python syntax check over `src` passed.
- `pytest tests` (no GPU, no model files): Torch/Diffusers are stubbed in `tests/conftest.py`, which also holds the recording Diffusers fakes (fixture `fake_diffusers`).
  - `test_generator_memory.py`: only a base model change unloads; LoRA / IP-Adapter requests are passed on; reference images are checked first; progress callbacks.
  - `test_adapter_updates.py`: incremental LoRA updates over a recording pipeline (weight-only change, add, remove, replace, unchanged, empty, model change, inline prompt LoRAs, failure clean-up).
  - `test_ip_adapter.py`: IP-Adapter load / scale / swap / unload / architecture check / offload hooks / failures over a recording pipeline, the reference images given to the pipeline, `load_image_encoder` single-file vs folder branches with stubbed libraries, the ViT-H/14 configuration against the h94 `config.json` fixture, and a subprocess with the real torch / transformers / diffusers (skipped when not installed) running the image encoder loader and `DiffusionRuntime` over a tiny random SD 1.x pipeline.
  - `test_checkpoint_inspection.py`: header parsing, SD 1.x / SDXL detection on tiny handcrafted `.safetensors` files, and `pipeline_architecture`.
  - `test_single_file_loading.py`: `ensure_model` picks the pipeline class from the detected family; shared scheduler configuration.
  - `test_textual_inversion.py`: single vs dual text-encoder registration; embedding vector resolution with real tensors (in a subprocess, skipped when torch is not installed).
  - `test_prompt_lora_syntax.py`: inline LoRA tag extraction and stripping, the lazy public API, and a subprocess check that `prompt_lora` imports without torch.
- Real image inference was not run here because no local GPU/model assets were provided in the environment.
