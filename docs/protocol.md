# Evaluation protocol

The v2 suite has 240 questions each from MMStar, BLINK, OCRBench, ChartQA, POPE and
MVBench. It retains all 144 v1 questions and adds 1,296 through deterministic,
output-independent stratified sampling. MVBench covers 19 publicly preparable tasks;
`fine_grained_pose` requires separately licensed NTU media and is excluded at suite
construction, not counted as model failure. Each video is represented by eight
chronological frames. There is no audio. Six no-image requests are diagnostic only;
warmup and diagnostics do not enter the headline score or latency distribution.

## Scoring

Every planned core image request counts, including failed, missing, empty and
unresolved responses. MCQ extraction prefers explicit final answers, answer boxes,
terminal choices and exact option text, without searching the gold answer inside
reasoning. Yes/no never defaults to yes on an unresolved answer. Native thought
blocks and terminal control tokens are removed consistently; formatting violations
are reported separately from correctness.

OCRBench follows substring matching (case-insensitive except handwritten formulas,
which remove spaces but preserve case). ChartQA uses 5% relative numeric tolerance;
zero must be zero, years and year lists use exact comparison. Unit conversion is
bound to question/input/reference/media hashes in packaged metadata and applied to
all models, never chosen according to a prediction. The final-answer adaptations
are project-specific, not byte-for-byte copies of the official evaluators. Source
references are recorded by `scoring-policy.json`, including pinned VLMEvalKit files.

Published source labels are not silently corrected. Optional historical human-review
scoring (`--review-manifest`) is a separate protocol and cannot be mixed into the
public-label radar chart. Unit metadata only applies where its binding matches;
new datasets require independently audited metadata, not borrowed hash entries.

## Timing and comparison

Report actual backend, precision, GPU count, thinking, visual processing and output
limits. API time includes network/provider overhead. Local total time includes
preprocessing and generation. TTFT is reported only when observed. A fast failed
request is not a successful answer; failures remain visible in completion/accuracy
counts. The radar speed axis is left blank unless all planned core requests succeeded
and have valid timings. For explicit recovery, cumulative observed
attempt times must be retained; waiting between runs is not request time.

The seven-axis chart uses six accuracies and `100*τ/(τ+median_seconds)` with τ=1 by
default. There is no radar-area leaderboard. Reusing one frozen suite and identical
scoring policy is necessary; it does not establish equal hardware or compute budgets.
Questions sharing images/videos are correlated. These stratified subsets cannot be
interpreted as complete official benchmark rankings.
