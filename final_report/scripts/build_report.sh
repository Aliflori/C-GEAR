#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repository_root=$(cd "${script_dir}/../.." && pwd)
python_bin=${PYTHON_BIN:-python}

cd "${repository_root}"
"${python_bin}" final_report/scripts/regenerate_six_seed_analysis.py
"${python_bin}" final_report/scripts/generate_report_figures.py

cd final_report/paper
pdflatex -interaction=nonstopmode -halt-on-error cgear_final_report.tex
bibtex cgear_final_report
pdflatex -interaction=nonstopmode -halt-on-error cgear_final_report.tex
pdflatex -interaction=nonstopmode -halt-on-error cgear_final_report.tex

if grep -Eq 'undefined references|Citation .* undefined|Overfull \\hbox|Overfull \\vbox' cgear_final_report.log; then
    echo "Report validation failed: unresolved reference/citation or overfull box." >&2
    exit 1
fi

page_count=$(mutool info cgear_final_report.pdf | awk '/^Pages:/ {print $2}')
if [ -z "${page_count}" ] || [ "${page_count}" -gt 4 ]; then
    echo "Report validation failed: expected at most four pages, found '${page_count:-unknown}'." >&2
    exit 1
fi

render_dir="${repository_root}/final_report/rendered/current"
mkdir -p "${render_dir}"
mutool draw -q -r 180 -o "${render_dir}/page-%d.png" cgear_final_report.pdf
echo "Built ${repository_root}/final_report/paper/cgear_final_report.pdf (${page_count} pages)."
echo "Rendered QA pages to ${render_dir}."
