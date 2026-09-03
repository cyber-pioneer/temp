# Qwen3.6-35B-A3B 4096/256 four-scenario comparison

## Scope

All four runs used tensor parallel size 2, 64 concurrent requests, 4096 input
tokens per request, and 256 output tokens per request. Each run completed one
64-request warmup batch before `/start_profile`; only the following 64-request
batch is present in the runtime trace. Extraction uses rank 0 only. CUDA Graph
construction and capture are not profiled.

The native runs used `VLLM_PLUGINS=""`. The API-server process environment was
checked and contained the explicit empty value. Their logs contained no
`Loading plugin fl` or `Platform plugin fl is activated` message. Both plugin
runs loaded and activated `fl`.

## Scenario summary

| Scenario | Runtime | Profiled batch (s) | Kernel events | Kernel types | Kernel duration (us) | CPU op types | Known shape/dtype variants | Operator-kernel relations | Numbered IDs | Event mapping | Time mapping | Rank-0 trace (MB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| plugin_graph | plugin-FL, CUDA Graph | 10.815 | 391,150 | 96 | 6,359,288.037 | 112 | 1,377 | 125 | 81 | 16.271% | 56.134% | 30.444 |
| plugin_eager | plugin-FL, eager | 95.867 | 896,162 | 104 | 9,624,662.593 | 102 | 3,863 | 105 | 41 | 99.871% | 99.966% | 191.447 |
| native_graph | native vLLM, CUDA Graph | 7.785 | 347,278 | 114 | 6,872,143.185 | 117 | 1,457 | 144 | 85 | 16.962% | 58.357% | 30.293 |
| native_eager | native vLLM, eager | 32.977 | 706,737 | 116 | 28,374,945.862 | 108 | 3,278 | 118 | 45 | 98.191% | 99.778% | 174.411 |

The profiled batch time includes profiler overhead and is not an unprofiled
throughput benchmark. Kernel duration is the sum of rank-0 runtime kernel
durations, not end-to-end latency.

Every conservation and classification check in all four `summary.json` files
is `true`. In particular, each run preserves the exact trace kernel count and
duration in both reports, retains kernels without logical metadata, and keeps
all operator-kernel relations unique.

## Operator classification

| Scenario | aten | custom | fused communication + compute | runtime operator | torch.compile | Triton compiled | unattributed | unattributed NVJet | pure communication |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| plugin_graph | 48 | 13 | 2 | 0 | 3 | 15 | 33 | 7 | 4 |
| plugin_eager | 79 | 17 | 0 | 2 | 3 | 0 | 1 | 0 | 3 |
| native_graph | 59 | 14 | 2 | 0 | 3 | 15 | 34 | 13 | 4 |
| native_eager | 90 | 18 | 0 | 0 | 3 | 0 | 4 | 0 | 3 |

Classification rules used by `operator_list.csv`:

- `moe_align_block_size_stage1`, `stage2_vec`, `stage3`, and `stage4` are
  normalized to `moe_align_block_size` and share one operator ID. Both plugin
  runs contain all four stages with ID 1.
- Except for that explicit stage family, `custom` rows use kernel-granularity
  identity. Different kernel names therefore receive different operator IDs,
  even when their source operator name is the same.
- Pure communication rows have `operator_id=null`, use
  `operator_kind=communication`, and are sorted after every numbered row.
- A fused operator that performs communication and computation, such as
  `vllm::flashinfer_trtllm_fused_allreduce_norm`, remains numbered and uses
  `operator_kind=fused_communication_compute`.
- Missing attribution is retained as `operator_name=null`; it is never removed
  merely because shape/dtype or an external ID is unavailable.

These rules are deterministic string and exact-name rules. They do not depend
on model-generated semantic classification.

## Type and shape differences

Each cell is `left / right / intersection / left-only / right-only`.

| Comparison | Kernel types | Raw attributed operator labels | Shape/dtype variants | Complete CPU op types |
|---|---|---|---|---|
| plugin graph vs native graph | 96 / 114 / 58 / 38 / 56 | 46 / 49 / 36 / 10 / 13 | 1,377 / 1,457 / 773 / 604 / 684 | 112 / 117 / 105 / 7 / 12 |
| plugin eager vs native eager | 104 / 116 / 65 / 39 / 51 | 35 / 36 / 24 / 11 / 12 | 3,863 / 3,278 / 2,750 / 1,113 / 528 | 102 / 108 / 96 / 6 / 12 |
| plugin graph vs plugin eager | 96 / 104 / 72 / 24 / 32 | 46 / 35 / 30 / 16 / 5 | 1,377 / 3,863 / 772 / 605 / 3,091 | 112 / 102 / 93 / 19 / 9 |
| native graph vs native eager | 114 / 116 / 94 / 20 / 22 | 49 / 36 / 33 / 16 / 3 | 1,457 / 3,278 / 1,284 / 173 / 1,994 | 117 / 108 / 99 / 18 / 9 |

Graph replay preserves complete physical kernel accounting but does not replay
most original CPU operator events. That is why graph runs have much lower
runtime logical shape attribution than eager runs. Plugin-FL also changes the
physical kernel program: plugin/native share only 58 kernel names in graph
mode and 65 in eager mode.

The complete unions and per-kernel percentage differences are in the generated
pairwise comparison CSV files; this document intentionally avoids copying long
demangled kernel names.

## Result locations

- `/vllm-workspace/graph_operator_profile_runs/qwen3_6_35b_a3b_plugin_graph_4096_256/`
- `/vllm-workspace/graph_operator_profile_runs/qwen3_6_35b_a3b_plugin_eager_4096_256/`
- `/vllm-workspace/graph_operator_profile_runs/qwen3_6_35b_a3b_baseline_graph_4096_256/`
- `/vllm-workspace/graph_operator_profile_runs/qwen3_6_35b_a3b_baseline_eager_4096_256/`

Pairwise comparisons are under this directory in `plugin_vs_native_graph`,
`plugin_vs_native_eager`, `plugin_graph_vs_eager`, and
`native_graph_vs_eager`. Each contains the complete kernel, operator,
shape/dtype, and CPU-operator comparisons.
