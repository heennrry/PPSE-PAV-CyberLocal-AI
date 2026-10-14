# CyberLocal-AI
CyberLocal AI, un asistente especializado en ciberseguridad. Mi objetivo es ayudarte en actividades legítimas de seguridad, como defensa, análisis, threat hunting, detección, respuesta a incidentes, análisis de malware, análisis forense, hardening, secure coding, auditorías, pentesting autorizado y simulaciones Red Team autorizadas.

Aplicación de IA local con FastAPI, Ollama y Nginx, desplegada con Docker Compose.

## Requisitos

- Docker Engine o Docker Desktop.
- Docker Compose v2.
- Conexión a Internet durante la primera instalación.
- Espacio libre suficiente para descargar el modelo de IA.
- Puerto 80 disponible para acceder a la interfaz web.

## Contenido del proyecto

- `app/main.py`: backend y aplicación web.
- `app/requirements.txt`: dependencias de Python.
- `Dockerfile`: construcción de la imagen de la aplicación.
- `docker-compose.yml`: servicios de Ollama, aplicación y Nginx.
- `nginx.conf`: configuración del proxy web.
- `documents/`: documentos que utiliza la aplicación.
- `install.sh`: instalador automatizado.
- `backups/`: carpeta local para copias de seguridad; no se incluye en Git.

## Instalación en un equipo nuevo

### 1. Instalar Docker

Instala Docker Engine y el complemento Docker Compose en Linux,
o Docker Desktop en Windows o macOS.

En Linux, asegúrate de que Docker está iniciado y de que tu usuario
puede ejecutar comandos Docker.

Comprueba la instalación:

```bash
docker --version
docker compose version
docker info

///Feliz Hacking :D
