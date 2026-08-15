#!/usr/bin/env bash
# Abbildungen bauen: TikZ -> PDF (fuers Paper) -> SVG (fuer die README).
#
# SVG statt PNG, weil PNG in der README sichtbar verpixelt. `pdftocairo -svg`
# wandelt jede Glyphe in einen Pfad um (0 <text>-Elemente im Ergebnis) — das
# Bild ist damit vektoriell UND schriftunabhaengig, rendert also ueberall
# identisch, ohne dass GitHub eine Schrift nachladen muesste.
#
# Palette und Node-Styles liegen in preamble.tex und sind identisch zum
# GermEval-2026-Paper, damit die Quellen dort ohne Neuzeichnen passen.
set -euo pipefail
cd "$(dirname "$0")"
for f in fig-scopes fig-methods fig-ensemble; do
    pdflatex -interaction=nonstopmode -halt-on-error "$f.tex" >/dev/null
    pdftocairo -svg "$f.pdf" "../$f.svg"
    printf '  %-14s -> ../%s.svg  (%s)\n' "$f.tex" "$f" \
        "$(wc -c <"../$f.svg" | tr -d ' ') B"
done
rm -f ./*.aux ./*.log
