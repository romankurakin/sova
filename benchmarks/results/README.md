# Benchmark results

`os-architectures-v1/` is the benchmark's data directory. `bench prepare`, `bench judge`,
`bench run` and `bench report` read and write it in place, so a fresh clone reuses the
committed work instead of regenerating it.

| Committed | Local only (`.gitignore`) |
| --- | --- |
| Markdown reading copies and their structure (`reading/`) | PDFs (`sources/`), downloaded and verified from [sources.lock.json](../sources.lock.json) |
| Passage snapshot (`passages/`) | Vectors (`embeddings/`) and the search index (`indexes/`), rebuilt by `bench prepare` in about 30 minutes |
| Generated contexts (`contexts/`) | Raw judge responses with token usage and timing (`judge-attempts/`) |
| Candidate pool, grade jobs and their queue (`pools/`, `jobs/`, `queue.json`) | Judge run state (`judge-state/`), locks |
| Judge answers and the resolved judge models (`judgments/`, `judge-runtimes/`) | The copy of the source lock |
| Reference revisions (`suites/`, `suite.json`) | Run checkpoints (`runs/<hash>/`), the experiment embedder log (`logs/`) |
| Compact runs, one result per run and the table (`runs/*.json`, `results/`, `report.json`, `EXPERIMENTS.md`) | |

## Contexts

All 3,473 contexts come from one model:

- **Weights:** [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF),
  file `Qwen3.8-27B-UD-Q4_K_XL.gguf`, SHA-256 `3f227079003add2511437e5b1e94812e363385225bf6a9b47b0054a72bc8b01e`.
- **Engine:** llama.cpp `llama-server`, builds `b11429-d81235049` (1,835 contexts) and
  `b11669-620d2225c` (1,638). Both builds produced byte-identical contexts where compared.
- **Server:** `--ctx-size 4096`, `--parallel 1`.
- **Request:** temperature 0, at most 512 tokens, reasoning effort `medium` with a
  256-token reasoning budget, a strict JSON schema with one `context` string. The prompt
  is `CONTEXT_SYSTEM_PROMPT` and `CONTEXT_USER_PROMPT` in
  [sova/llama_client.py](../../sova/llama_client.py): document name, section path, the
  passage, and 500 characters of each neighbor within the same section.

Every context file records the weights' hash, the chat template hash, the exact request,
the llama.cpp build and the model name above.

## Vectors

Rebuilt locally from the contexts: [Qwen/Qwen3-Embedding-4B-GGUF](https://huggingface.co/Qwen/Qwen3-Embedding-4B-GGUF),
file `Qwen3-Embedding-4B-Q8_0.gguf`, SHA-256 `b60ae5ce2dd6a0b77f82cadf21def1f310a3e10cde380ad0081b07a9d416949d`,
last-token pooling, 2,560 dimensions, `--ctx-size 4096`.

## Judges

Models and effort are in [judges.json](../judges.json). Each answer records the judge's
CLI version and, for Claude, the model that actually answered. Grades are model-generated;
no person checked them.

## Sources and licenses

The Markdown, passages and contexts are derived from these documents and keep their
licenses; attribution, changes made and the full xv6 license are in
[NOTICE.md](NOTICE.md). Exact files, URLs and hashes are in
[sources.lock.json](../sources.lock.json).

| Source | Version | License |
| --- | --- | --- |
| RISC-V Unprivileged ISA | 20250508 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| RISC-V Privileged Architecture | 20250508 | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| LoongArch Reference Manual, Volume 1: Basic Architecture | 1.10 | [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) |
| xv6: a simple, Unix-like teaching operating system (RISC-V) | rev5 | [MIT-style book license](https://github.com/mit-pdos/xv6-riscv-book/blob/xv6-riscv/LICENSE) |
| Max Hailperin, Operating Systems and Middleware: Supporting Controlled Interaction | Revised Edition 1.3.1 | [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) |
