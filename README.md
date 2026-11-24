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
- `schemas.py` — `ModelSpec`, `LoRASpec`, `KSampleConfig`, `GenerateResult`, and path helpers.
- `checkpoint_inspection.py` — reads a `.safetensors` header (pure Python, no torch) to tell SD 1.x from SDXL single-file checkpoints.

The package's public names are resolved lazily (PEP 562): `import diffusion_lora_runtime` and its torch-free modules (`prompt_lora`, `normalization`, `prompt_assets`, `schemas`, `checkpoint_inspection`) never import torch or diffusers; they load the first time a name that needs them (`DiffusionGenerator`, `AdapterLoadError`, `GenerationCancelled`, `PromptEmbeddingResolutionError`) is accessed.

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

Shortcut overrides are also accepted: `width`, `height`, `steps`, `cfg`, and `seed`. They override the values inside `ksample` for that call.

### `DiffusionGenerator.unload()`

Cancels any current generation flag, drops the Diffusers pipeline, clears adapter/textual-inversion bookkeeping, runs `gc.collect()`, and empties the CUDA cache when CUDA is available.

### Checkpoint inspection

```python
from diffusion_lora_runtime import detect_single_file_architecture, read_safetensors_keys

detect_single_file_architecture("realisticVision.safetensors")  # "sd15"
detect_single_file_architecture("illustriousXL.safetensors")    # "sdxl"
detect_single_file_architecture("model.ckpt")                   # None (never parsed)
read_safetensors_keys("model.safetensors")                       # tensor names, header order
```

Both only read the safetensors header (8-byte little-endian length + JSON); no tensor is loaded and torch is not imported. `read_safetensors_keys` raises `SafetensorsHeaderError` for malformed headers; `detect_single_file_architecture` returns `None` for anything it cannot classify (see "Single-file checkpoints" below).


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

`generate()` remembers the loaded model and LoRAs.

- Same model path + dtype + same resolved LoRA paths + same LoRA weights: reuse the already-loaded pipeline/adapters.
- Different model path, dtype, LoRA path, or LoRA weight: unload the previous runtime first, then load the new signature.
- Explicit LoRAs and inline prompt LoRAs are combined by resolved path. Inline tags can override a duplicate explicit LoRA's weight because this matches the original backend flow.
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
- `pytest tests` (no GPU, no model files): Torch/Diffusers are stubbed in `tests/conftest.py`.
  - `test_generator_memory.py`: model/LoRA reuse and progress callbacks.
  - `test_checkpoint_inspection.py`: header parsing and SD 1.x / SDXL detection on tiny handcrafted `.safetensors` files.
  - `test_single_file_loading.py`: `ensure_model` picks the pipeline class from the detected family; shared scheduler configuration.
  - `test_textual_inversion.py`: single vs dual text-encoder registration; embedding vector resolution with real tensors (in a subprocess, skipped when torch is not installed).
  - `test_prompt_lora_syntax.py`: inline LoRA tag extraction and stripping, the lazy public API, and a subprocess check that `prompt_lora` imports without torch.
- Real image inference was not run here because no local GPU/model assets were provided in the environment.
