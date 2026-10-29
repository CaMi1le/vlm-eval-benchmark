# Contributing

Use a separate Python environment and run `python -m pytest -q` before proposing changes.
The test suite uses generated images, small synthetic fixtures, and local HTTP servers;
it must not invoke real model APIs or download weights.

Keep answer extraction independent of reference answers. Preserve planned failures in
score denominators, and distinguish runtime errors from parsing errors. A changed scoring
policy must update its recorded policy and regression fixtures; never rewrite archived
results to conceal changes. Retain exact model revisions, runtime versions, budgets and
input hashes in any published comparison.

Do not commit credentials, private endpoints, machine-specific configurations, benchmark
media, weights or full runtime outputs. Use `*.local.json` for local credentials/configs.
Add generic examples and small original fixtures instead. Before publication run
`python scripts/check_public_tree.py` against the tracked repository.

New adapters should use existing frozen cases and return response/timing metadata; see
`docs/data-format.md`. GPU validation is separate from CPU CI and must report actual
hardware and framework versions.

Optional UI checks need Node 22+ and a local Chromium. Set `CHROMIUM_BIN` if it is
not `/snap/bin/chromium`. After generating a demo or comparison, run:

```bash
node scripts/check_dashboard_browser.mjs demo-output/report/dashboard/dashboard.html reports/browser-radar 2
node scripts/check_report_browser.mjs demo-output/report/review.html reports/browser-gallery.json 2
```

The final argument is the expected model count. Browser checks block external DNS
and exercise model selection, filtering, images, exports and responsive layout.
