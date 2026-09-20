# Changelog

All notable changes to the Mara project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.0] - 2026-09-20

### Added
- **Prefill-Only Decision Transformer Architecture**: Bilinear PointerHead for sub-5ms single-pass neural tool routing on CPUs and microcontrollers.
- **AFM Tool Calling Engine (`mara.afm`)**: Python `@tool` decorator generating JSON Schemas from type hints with automatic TypeSafe fast-path dispatch.
- **Multi-Step Tool Planner (`mara.planner`)**: Automated decomposition of compound sentences and high-level macro routines into ordered execution graphs.
- **Hardware Abstraction Primitives**: Digital GPIO read/write, PWM duty cycles, and bus telemetry drivers.
- **Hugging Face Hub Integration**: Support for loading `model.safetensors`, `config.json`, and HF BPE tokenizers directly from `jaswanthsanjay88/mara`.
- **Monochrome Tool Playground v2**: Interactive Web interface with top-view SVG floor plan, light falloff shaders, rotating fan animations, and ghost previews.

### Changed
- Refactored routing pipeline to eliminate autoregressive generation latency for deterministic tool dispatch.
- Compressed model checkpoint size to 2.7 MB (`mara_afm.pt`) and 3.3 MB (`model.safetensors`).

---

## [0.1.0] - 2026-08-26

### Added
- Initial proof-of-concept for small-model intent parsing and browser-based WebGPU/WASM inference.
- Custom 8k byte-level BPE tokenizer.
