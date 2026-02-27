#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_FILE="${REPO_ROOT}/ollama_models.txt"

if [[ ! -f "${MODELS_FILE}" ]]; then
  echo "ERROR: ${MODELS_FILE} not found"
  exit 1
fi

if ! command -v ollama >/dev/null 2>&1; then
  echo "ERROR: ollama is not installed."
  echo "Install it from https://ollama.com/ then re-run: make setup-models"
  exit 1
fi

echo "Using models list: ${MODELS_FILE}"

while IFS= read -r model || [[ -n "${model}" ]]; do
  model="${model%%#*}"
  model="$(echo -n "${model}" | xargs)"
  if [[ -z "${model}" ]]; then
    continue
  fi
  echo "Pulling: ${model}"
  ollama pull "${model}"
done < "${MODELS_FILE}"

echo "Done. Installed models:"
ollama list
