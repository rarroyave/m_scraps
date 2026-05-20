#!/bin/bash
# Build docs/OVERVIEW.{tex,pdf} from docs/OVERVIEW.md via pandoc + lualatex.
#
# Requires:
#   - pandoc 2.x or 3.x
#   - lualatex (TeX Live, MacTeX, or MiKTeX)
#   - STIX Two Text + STIX Two Math fonts (system-installed; STIX is bundled
#     with macOS and most Linux TeX Live installs)
#   - Menlo for monospace (macOS default; substitute "DejaVu Sans Mono" on
#     Linux if needed)
set -e

cd "$(dirname "$0")"

# Preprocess to escape two Unicode chars that STIX Two doesn't include
# (->, infty); send to a tmp file rather than touching the source markdown.
SRC=$(mktemp /tmp/overview_pp.XXXXXX.md)
sed -e 's/→/$\\to$/g' -e 's/∞/$\\infty$/g' OVERVIEW.md > "$SRC"

COMMON_ARGS=(
    --from=markdown+pipe_tables+tex_math_dollars+raw_tex
    --to=latex
    --standalone
    -V documentclass=article
    -V geometry:margin=1in
    -V fontsize=11pt
    -V mainfont="STIX Two Text"
    -V mathfont="STIX Two Math"
    -V monofont="Menlo"
    -V colorlinks=true
    -V linkcolor=blue
    -V urlcolor=blue
    -V title="scraps-perpair: project overview"
    -V author="Raymundo Arroyave"
    -V date="$(date +%Y-%m-%d)"
    --toc --toc-depth=2
)

echo "Building OVERVIEW.tex..."
pandoc "$SRC" "${COMMON_ARGS[@]}" -o OVERVIEW.tex

echo "Building OVERVIEW.pdf (lualatex)..."
pandoc "$SRC" "${COMMON_ARGS[@]}" --pdf-engine=lualatex -o OVERVIEW.pdf

rm "$SRC"
echo "Done. Generated:"
ls -la OVERVIEW.tex OVERVIEW.pdf
