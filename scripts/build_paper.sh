#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../paper"
(cd figures && latexmk -pdf -interaction=nonstopmode speechmaster_overview.tex)
latexmk -pdf -interaction=nonstopmode main.tex
