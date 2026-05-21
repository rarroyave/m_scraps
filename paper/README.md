# JOSS paper

This directory contains the [JOSS](https://joss.theoj.org/)
submission for `scraps-perpair`:

- `paper.md` — the manuscript (Markdown with YAML front matter, JOSS
  convention). Target length is 250-1000 words for the summary +
  statement of need + algorithmic contributions + validation.
- `paper.bib` — BibTeX references.

## Local preview

JOSS papers compile via [Whedon](https://github.com/openjournals/whedon)
or [Openjournals/inara](https://github.com/openjournals/inara), but
for local previewing you can use pandoc:

```bash
pandoc paper.md -o paper.pdf --bibliography paper.bib --pdf-engine lualatex \
    -V geometry:margin=1in -V mainfont="STIX Two Text" -V mathfont="STIX Two Math"
```

The published rendering uses JOSS's house style; the local PDF is for
proofreading only.

## Submission checklist

When ready to submit:

1. Verify the `orcid` field in `paper.md` matches the author's record.
2. Update the `date` field to the submission date.
3. Confirm the repository is publicly accessible (JOSS requires
   public repos at review time -- the `rarroyave/scraps-perpair`
   fork would need to be made public before submission).
4. Tag a release version that the submission cites (suggest the
   current `v0.2.2` or a fresh `v0.3.0` post-warm-start).
5. Submit via [https://joss.theoj.org/papers/new](https://joss.theoj.org/papers/new)
   with the repository URL and the commit hash being reviewed.

## What JOSS reviewers will check

- The software has a clear, well-documented purpose.
- Installation instructions work on a fresh clone.
- Substantial functionality (not a trivial wrapper).
- Quality automated tests with non-trivial coverage.
- Documentation: README, API/CLI help, statement of need.
- Community guidelines: CONTRIBUTING.md, issue tracker.
- The paper itself is well-written and references the field.

All of the above are in place as of v0.2.2 (60+ tests, comprehensive
README, OVERVIEW.md, CONTRIBUTING.md, tutorials, mcsqs validation).
