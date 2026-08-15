#!/usr/bin/env bash
# Abbildungen bauen: TikZ -> PDF (fuers Paper) -> PNG (fuer die README).
# Die Quellen teilen preamble.tex und benutzen die Palette des
# GermEval-2026-Papers, damit sie dort ohne Neuzeichnen wiederverwendbar sind.
set -euo pipefail
cd "$(dirname "$0")"
for f in fig-scopes fig-methods fig-ensemble; do
    pdflatex -interaction=nonstopmode -halt-on-error "$f.tex" >/dev/null
    pdftocairo -png -r 200 -singlefile "$f.pdf" "../$f"
    echo "  $f.tex -> ../$f.png"
done
rm -f ./*.aux ./*.log
