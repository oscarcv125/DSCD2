#!/usr/bin/env bash
set -euo pipefail

# Sirve el frontend en su propio puerto para dejar claro que es un cliente
# separado de la API. Tambien funciona abriendo http://127.0.0.1:8000
# porque la API sirve el mismo index.html.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "$script_dir/.." && pwd)"

cd "$project_dir/frontend"
python -m http.server 5500
