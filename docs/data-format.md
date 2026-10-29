# Frozen inputs and result directories

A suite contains `cases.jsonl`, `references.jsonl`, `views.json`, `freeze.json` and
relative media files. `freeze.json` lists every input file's path, byte count and
SHA256. `vlmbench verify` detects modified files and input/media bindings.

Each case contains `id`, `dataset`, `category`, `task` (`mcq`, `yesno`, `ocr`, `chart`),
`question`, `choices`, `prompt`, `media` and `input_sha256`. Media entries contain
`path`, `sha256`, `width`, `height` and optional `timestamp_seconds`. Input hashes
bind question, choices, media, task and prompt. Gold `answers` are kept separately
in `references.jsonl` and never passed to model adapters.

`views.json` contains case ID lists for `core`, `smoke`, `regression`, `delta` and
`no_image_diagnostic`. Dataset preparation produces the standard six-axis suite.
Custom suites can be scored independently; the radar currently requires all six
standard datasets and the report gallery expects full core coverage. `vlmbench demo`
is an executable, entirely synthetic reference for the complete format.

A result directory contains:

- `run.json`: model identifier, frozen suite SHA256, effective config and observed metadata.
- `planned.json`: all `(id, condition, input_sha256)` requests.
- `outputs.jsonl`: one outcome per planned request, including failures. Fields include
  `id`, `model`, `condition` (`image` or `no_image`), `input_sha256`, `status`,
  `answer`, and observed `total_seconds`; token/timing details are optional.

Missing or inconsistent identities/hashes are rejected. Incomplete output sets are rejected by scoring. Runners record missing outcomes
as explicit failures rather than silently reducing the denominator. Do not use
`status=ok` for an exception. Preserve raw response, final-answer surface, finish
reason, and observed token counts when available; provider reasoning is not graded.

`score` emits model summaries, per-case grades, CSV, parser evidence and scoring
policy. `report` adds images and an interactive review page. `radar` reads completed
comparison summaries and snapshots their exact bytes alongside portable HTML;
different suite/policy/denominator hashes cannot be silently combined.
