#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")"

MODEL="dagbs/qwen2.5-coder-7b-instruct-abliterated:q4_k_m"

if ! docker info >/dev/null 2>&1; then
    echo "ERROR: Docker no está iniciado o tu usuario no tiene permisos."
    exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
    echo "ERROR: necesitas Docker Compose v2."
    exit 1
fi

mkdir -p documents

echo "==> Construyendo la aplicación e iniciando servicios..."
docker compose up -d --build

echo "==> Esperando a que Ollama esté disponible..."
ready=false
for i in $(seq 1 90); do
    if docker compose exec -T ollama ollama list >/dev/null 2>&1; then
        ready=true
        break
    fi
    sleep 2
done

if [ "$ready" != true ]; then
    echo "ERROR: Ollama no respondió. Revisa: docker compose logs ollama"
    exit 1
fi

echo "==> Descargando o verificando el modelo: $MODEL"
docker compose exec -T ollama ollama pull "$MODEL"

echo "==> Estado final:"
docker compose ps

echo
echo "Instalación terminada. Abre http://localhost"
echo "Desde otro dispositivo, usa la IP local de este ordenador."
