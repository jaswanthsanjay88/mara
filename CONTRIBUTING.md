# Contributing to Mara

Thank you for your interest in contributing to Mara! We welcome contributions that improve tool-calling efficiency, microcontroller runtime support, Model Context Protocol (MCP) integrations, and hardware abstraction drivers.

---

## Code of Conduct

Please maintain a collaborative, welcoming, and professional environment. Focus on constructive feedback and evidence-based technical discussions.

---

## Development Setup

1. **Fork and Clone**:
   ```bash
   git clone https://github.com/jaswanthsanjay88/mara.git
   cd mara
   ```

2. **Environment & Dependencies**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Running the Benchmarks & Tests**:
   ```bash
   python research/benchmark_afm.py --num-samples 100
   python -m unittest discover tests
   ```

---

## Contribution Guidelines

1. **Sub-5ms Inference Budget**:
   Any architecture or routing modification must respect edge hardware constraints. Do not introduce unbounded loops or allocations in the fast-path decision pipeline.

2. **Zero-Syntax-Error Discipline**:
   Mara enforces deterministic pointer projection and structured execution. PRs modifying tool dispatch must ensure syntax error rates remain 0.0% guaranteed.

3. **Typed Tools**:
   All new tool primitives must provide complete Python type annotations and docstrings for automated schema derivation.

4. **Testing**:
   Add test coverage under `tests/` or runnable examples under `examples/` for any new tool definitions, hardware registers, or planner routines.

---

## Submitting a Pull Request

1. Create a feature branch: `git checkout -b feature/your-feature-name`.
2. Commit your changes with concise, descriptive commit messages.
3. Verify that `benchmark_afm.py` passes with zero regressions.
4. Push to your fork and submit a PR to `main`.
