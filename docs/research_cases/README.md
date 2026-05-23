# Research case studies

End-to-end SCRAPs workflows for real materials-science design
problems. Each case study is a self-contained markdown + script pair
that you can run from the repo root.

## Cases

| # | Title | System | Key capability demonstrated |
|---:|---|---|---|
| 01 | [CoCrNi CSRO from STEM](01_cocrni_csro.md) | NiCoCr MEA on FCC | per-pair α targeting + Warren-Cowley sum rule for same-species constraints |

## Running a case

```bash
python3 docs/research_cases/run_<case>.py
```

Each script generates outputs into `docs/research_cases/results/<case>/`
(gitignored) — the markdown writeup embeds the resulting tables and
figures inline so the case study is self-contained even without
re-running.

## Adding a new case

The minimal pattern is a Python script that orchestrates SCRAPs
through `scraps_ase.run_sqs_from_atoms`, captures the timing and
verified α matrix, and writes a JSON summary + a matplotlib figure;
plus a markdown writeup explaining what was done, why, and what the
result means. Use `01_cocrni_csro` as a template.
