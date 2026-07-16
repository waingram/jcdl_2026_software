# src/config.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml

from .prompting import SDGSpec, Variant


@dataclass(frozen=True)
class ModelConfig:
    name: str
    trust_remote_code: bool = True
    device_map: str = "auto"
    torch_dtype: str = "auto"
    seed: int = 0
    padding_side: str = "left"
    ensure_pad_token: bool = True


@dataclass(frozen=True)
class InputConfig:
    path: Path
    doi_col: str = "DOI"
    text_col: str = "Abstract"


@dataclass(frozen=True)
class OutputConfig:
    dir: Path
    basename: str
    write_debug_raw: bool = False


@dataclass(frozen=True)
class RunSection:
    sdg_number: str
    batch_size: int
    max_rows: int | None = None
    resume: bool = False


@dataclass(frozen=True)
class PromptConfig:
    variant: Variant = "binary_label"
    include_contribution_types: bool = True
    include_indirect_clause: bool = True


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 3
    return_full_text: bool = False
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float | None = None
    top_k: int | None = None


@dataclass(frozen=True)
class RunConfig:
    version: int
    sdgs_path: Path
    model: ModelConfig
    input: InputConfig
    output: OutputConfig
    run: RunSection
    prompt: PromptConfig
    generation: GenerationConfig

    @property
    def output_labels_csv(self) -> Path:
        # (DOI, SDG, label)
        fname = f"{self.output.basename}.csv"
        return self.output.dir / fname

    @property
    def output_raw_csv(self) -> Path:
        # optional raw generated_text for debugging
        fname = f"{self.output.basename}_raw.csv"
        return self.output.dir / fname


def load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_sdgs(path: Path) -> Dict[str, SDGSpec]:
    doc = load_yaml(path)
    if not isinstance(doc, Mapping):
        raise ValueError("sdgs.yaml must be a mapping at the top level")

    sdgs_raw = doc.get("sdgs")
    if sdgs_raw is None:
        raise ValueError("sdgs.yaml must contain key 'sdgs'")
    if not isinstance(sdgs_raw, Mapping):
        raise ValueError("sdgs.yaml['sdgs'] must be a mapping of sdg_number -> spec")

    out: Dict[str, SDGSpec] = {}
    for k, v in sdgs_raw.items():
        number = str(k).strip()
        if not isinstance(v, Mapping):
            raise ValueError(f"sdgs[{number!r}] must be a mapping")

        title = str(v.get("title", "")).strip()
        description = str(v.get("description", "")).strip()

        targets_val = v.get("targets_md", None)
        if targets_val is None:
            targets_val = v.get("targets", "")

        targets = str(targets_val or "").strip()
        if not number or not title or not targets:
            raise ValueError(
                f"SDG entry missing required fields for {number!r}: {dict(v)}"
            )

        out[number] = SDGSpec(
            number=number, title=title, description=description, targets=targets
        )

    return out


def load_run_config(path: Path) -> RunConfig:
    doc = load_yaml(path)
    if not isinstance(doc, Mapping):
        raise ValueError("run config must be a mapping")

    def _as_map(v: Any, k: str) -> Mapping[str, Any]:
        if not isinstance(v, Mapping):
            raise ValueError(f"Missing or invalid mapping for key {k!r}")
        return v

    def _req_str(m: Mapping[str, Any], k: str) -> str:
        v = m.get(k)
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"Missing or invalid string key {k!r}")
        return v.strip()

    def _opt_str(m: Mapping[str, Any], k: str, default: str) -> str:
        v = m.get(k, default)
        if v is None:
            return default
        if not isinstance(v, str):
            raise ValueError(f"Invalid string key {k!r}")
        return v.strip()

    def _req_int(m: Mapping[str, Any], k: str) -> int:
        v = m.get(k)
        if not isinstance(v, int) or v <= 0:
            raise ValueError(f"Missing or invalid positive int key {k!r}")
        return v

    def _opt_int(m: Mapping[str, Any], k: str, default: int | None) -> int | None:
        v = m.get(k, default)
        if v is None:
            return None
        if not isinstance(v, int):
            raise ValueError(f"Invalid int key {k!r}")
        return v

    def _opt_bool(m: Mapping[str, Any], k: str, default: bool) -> bool:
        v = m.get(k, default)
        if not isinstance(v, bool):
            raise ValueError(f"Invalid bool key {k!r}")
        return v

    def _opt_float(m: Mapping[str, Any], k: str, default: float | None) -> float | None:
        v = m.get(k, default)
        if v is None:
            return None
        if not isinstance(v, (int, float)):
            raise ValueError(f"Invalid float key {k!r}")
        return float(v)

    version = doc.get("version", 1)
    if not isinstance(version, int) or version <= 0:
        raise ValueError("version must be a positive int")

    sdgs_path = Path(_req_str(doc, "sdgs_path"))
    model_m = _as_map(doc.get("model"), "model")
    input_m = _as_map(doc.get("input"), "input")
    output_m = _as_map(doc.get("output"), "output")
    run_m = _as_map(doc.get("run"), "run")
    prompt_m = _as_map(doc.get("prompt", {}), "prompt")
    gen_m = _as_map(doc.get("generation", {}), "generation")

    model = ModelConfig(
        name=_req_str(model_m, "name"),
        trust_remote_code=_opt_bool(model_m, "trust_remote_code", True),
        device_map=_opt_str(model_m, "device_map", "auto"),
        torch_dtype=_opt_str(model_m, "torch_dtype", "auto"),
        seed=_opt_int(model_m, "seed", 0) or 0,
        padding_side=_opt_str(model_m, "padding_side", "left"),
        ensure_pad_token=_opt_bool(model_m, "ensure_pad_token", True),
    )

    inp = InputConfig(
        path=Path(_req_str(input_m, "path")),
        doi_col=_opt_str(input_m, "doi_col", "DOI"),
        text_col=_opt_str(input_m, "text_col", "Abstract"),
    )

    out = OutputConfig(
        dir=Path(_req_str(output_m, "dir")),
        basename=_req_str(output_m, "basename"),
        write_debug_raw=_opt_bool(output_m, "write_debug_raw", False),
    )

    run = RunSection(
        sdg_number=_req_str(run_m, "sdg_number"),
        batch_size=_req_int(run_m, "batch_size"),
        max_rows=_opt_int(run_m, "max_rows", None),
        resume=_opt_bool(run_m, "resume", False),
    )

    prompt = PromptConfig(
        variant=_opt_str(prompt_m, "variant", "binary_label"),  # type: ignore[arg-type]
        include_contribution_types=_opt_bool(
            prompt_m, "include_contribution_types", True
        ),
        include_indirect_clause=_opt_bool(prompt_m, "include_indirect_clause", True),
    )

    generation = GenerationConfig(
        max_new_tokens=_opt_int(gen_m, "max_new_tokens", 3) or 3,
        return_full_text=_opt_bool(gen_m, "return_full_text", False),
        do_sample=_opt_bool(gen_m, "do_sample", False),
        temperature=float(gen_m.get("temperature", 0.0) or 0.0),
        top_p=_opt_float(gen_m, "top_p", None),
        top_k=_opt_int(gen_m, "top_k", None),
    )

    return RunConfig(
        version=version,
        sdgs_path=sdgs_path,
        model=model,
        input=inp,
        output=out,
        run=run,
        prompt=prompt,
        generation=generation,
    )
