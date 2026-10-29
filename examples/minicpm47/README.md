# Archived MiniCPM-V 4.7 comparison

These are real aggregated results from October 2026, not the synthetic CPU demo.
The public `comparison.json` preserves recorded scores, timings, case IDs and source
hashes, with machine paths/hostnames removed from execution context. `provenance.json`
records the original summary hash. Raw answers and source benchmark media remain
outside this repository. These aggregates alone cannot independently verify every
original prediction.

| Dataset (240 each) | BF16 | INT4 AWQ |
|---|---:|---:|
| MMStar | 184 | 181 |
| BLINK | 161 | 154 |
| OCRBench | 217 | 212 |
| ChartQA | 211 | 213 |
| POPE | 209 | 208 |
| MVBench | 171 | 174 |
| Total | **1153 / 1440 (80.07%)** | **1142 / 1440 (79.31%)** |

Both have 1,440 image outcomes and six separate no-image diagnostics. AWQ used one
48 GiB RTX 4090 (reported by the host), vLLM 0.32.0 / Transformers 5.18.0; BF16 used
two GPUs and Transformers. Both used nonthinking greedy decoding, 131,072 context,
natural EOS output, 16x compression, single-image four slices / multi-image one slice,
and eight sampled frames per video. AWQ checks verified 40 quantized expert layers.

AWQ median complete response was 0.272 seconds; BF16 was 0.399 seconds. BF16 reuses
previous complete responses plus six predetermined wider-output supplements; its
median measures selected complete responses, not cumulative original truncated
attempt cost. AWQ has no such supplements. The hardware/runtime and response
selection differences prevent treating this as a pure quantization ablation or
same-hardware acceleration comparison. AWQ scored 11 fewer questions (-0.76 points).

Generate the interactive offline chart from the repository root:

```bash
vlmbench radar --scores examples/minicpm47/comparison.json --output reports/minicpm47
```

Open `reports/minicpm47/dashboard.html`. The archived scoring policy/code hashes
remain historical; this file has not been relabeled as a new score from the public
package. Newly prepared suites can have different freeze hashes because preparation
records environment/timestamps. Packaged chart-unit metadata requires its exact frozen
suite plus per-input/reference/media hashes; it is not automatically transferred to
new freezes. Use identical frozen inputs and explicit policy when comparing new runs.
