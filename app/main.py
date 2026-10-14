import os
import re
import json
from pathlib import Path
from typing import Optional

import requests
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel


# ============================================================
# CONFIG
# ============================================================

APP_NAME = "CyberLocal AI"

MODEL = os.getenv(
    "OLLAMA_MODEL",
    "dagbs/qwen2.5-coder-7b-instruct-abliterated:q4_k_m",
)

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://ollama:11434/api/generate",
)

DOCUMENTS_DIR = Path(
    os.getenv("DOCUMENTS_DIR", "/documents")
)

DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)


app = FastAPI(
    title=APP_NAME,
    version="2.0.0",
)


# ============================================================
# MODELOS
# ============================================================

class AskRequest(BaseModel):
    question: str
    conversation: Optional[list] = None
    mode: Optional[str] = "blue"
    target: Optional[str] = ""
    context: Optional[str] = ""


# ============================================================
# SYSTEM PROMPTS
# ============================================================

BASE_SYSTEM_PROMPT = """
Eres CyberLocal AI, un asistente local especializado en ciberseguridad.

Responde en español salvo que el usuario solicite otro idioma.

Tu objetivo es ayudar en actividades legítimas de seguridad:
- defensa
- análisis
- threat hunting
- detección
- respuesta a incidentes
- análisis de malware
- análisis forense
- hardening
- secure coding
- auditorías
- pentesting autorizado
- simulaciones Red Team autorizadas
- validación de controles de seguridad
- documentación técnica.

Debes distinguir siempre entre:
1. información técnica educativa,
2. pruebas autorizadas,
3. acciones potencialmente destructivas.

Cuando una operación pueda afectar sistemas reales, exige que el contexto sea autorizado y recomienda trabajar primero sobre laboratorio, staging, CTF o sistemas explícitamente autorizados.

No inventes resultados de comandos, escaneos, logs, exploits ni evidencias que no hayan sido proporcionados.

Si faltan datos, dilo claramente.

Cuando sea útil:
- explica el razonamiento técnico de alto nivel,
- proporciona comandos de diagnóstico seguros,
- propone pasos reproducibles,
- indica qué evidencia buscar,
- explica cómo validar el resultado,
- incluye mitigaciones.

Prioriza precisión, trazabilidad y seguridad operacional.
"""


RED_TEAM_PROMPT = """
MODO RED TEAM

Actúas como asistente de Red Team para ejercicios autorizados.

Áreas principales:
- reconnaissance
- attack surface mapping
- vulnerability assessment
- web security testing
- API security
- authentication testing
- authorization testing
- Active Directory security assessment
- privilege escalation assessment
- lateral movement assessment
- credential exposure analysis
- phishing simulation planning
- payload analysis en laboratorio
- C2 simulation conceptual
- persistence assessment
- evasion assessment
- attack-path analysis
- adversary emulation
- purple-team exercises.

Trabaja con mentalidad de atacante, pero dentro de un entorno autorizado.

Para cada ejercicio intenta estructurar:
1. Objetivo
2. Supuestos
3. Alcance
4. Reconocimiento
5. Hipótesis
6. Técnica de validación
7. Evidencia esperada
8. Riesgo
9. Detecciones que debería generar
10. Mitigación

No afirmes que una vulnerabilidad existe sin evidencia.

Cuando falte alcance, pide o identifica claramente qué parte del entorno debe estar autorizada.
"""


BLUE_TEAM_PROMPT = """
MODO BLUE TEAM

Actúas como especialista defensivo.

Áreas principales:
- SOC
- SIEM
- threat hunting
- incident response
- malware triage
- IOC analysis
- log analysis
- Windows security
- Linux security
- Active Directory defense
- EDR
- network detection
- DNS security
- email security
- cloud security
- vulnerability management
- hardening
- detection engineering
- MITRE ATT&CK
- forensic analysis
- containment
- eradication
- recovery.

Prioriza:
1. detección
2. evidencia
3. triage
4. impacto
5. contención
6. erradicación
7. recuperación
8. prevención.

Cuando sea útil, proporciona:
- queries SIEM,
- reglas de detección,
- Sigma conceptual,
- YARA para análisis defensivo,
- indicadores,
- hipótesis de hunting,
- timeline,
- checklist de respuesta,
- recomendaciones de hardening.

No inventes eventos ni evidencias.
"""


PURPLE_TEAM_PROMPT = """
MODO PURPLE TEAM

Combinas Red Team y Blue Team.

Para cada técnica ofensiva:
1. explica el objetivo de la técnica,
2. describe cómo validar de forma autorizada,
3. identifica la telemetría que debería producir,
4. propone detecciones,
5. explica cómo comprobar si la defensa funciona,
6. recomienda mitigaciones.

El objetivo es mejorar conjuntamente ataque y defensa.
"""


# ============================================================
# RAG / DOCUMENTOS
# ============================================================

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".md",
    ".log",
    ".csv",
    ".json",
    ".xml",
    ".yaml",
    ".yml",
}


def extract_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages = []

        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                continue

        return "\n".join(pages)

    except Exception as exc:
        return f"[Error leyendo PDF: {exc}]"


def read_document(path: Path) -> str:
    extension = path.suffix.lower()

    if extension == ".pdf":
        return extract_pdf(path)

    try:
        return path.read_text(
            encoding="utf-8",
            errors="ignore",
        )
    except Exception as exc:
        return f"[Error leyendo documento: {exc}]"


def tokenize(text: str):
    return set(
        re.findall(
            r"[a-zA-ZÀ-ÿ0-9_./:-]{3,}",
            text.lower(),
        )
    )


def search_documents(
    question: str,
    max_documents: int = 5,
    max_chars: int = 12000,
):
    question_tokens = tokenize(question)

    if not question_tokens:
        return []

    candidates = []

    for path in DOCUMENTS_DIR.iterdir():
        if not path.is_file():
            continue

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        try:
            content = read_document(path)

            if not content.strip():
                continue

            content_tokens = tokenize(content)

            score = len(
                question_tokens.intersection(content_tokens)
            )

            if score > 0:
                candidates.append(
                    (
                        score,
                        path.name,
                        content,
                    )
                )

        except Exception:
            continue

    candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    results = []

    total_chars = 0

    for score, name, content in candidates[:max_documents]:
        remaining = max_chars - total_chars

        if remaining <= 0:
            break

        excerpt = content[:remaining]

        results.append(
            {
                "name": name,
                "score": score,
                "content": excerpt,
            }
        )

        total_chars += len(excerpt)

    return results


# ============================================================
# OLLAMA
# ============================================================

def get_mode_prompt(mode: str) -> str:
    mode = (mode or "blue").lower()

    if mode == "red":
        return RED_TEAM_PROMPT

    if mode == "purple":
        return PURPLE_TEAM_PROMPT

    return BLUE_TEAM_PROMPT


def build_prompt(
    question: str,
    conversation=None,
    mode="blue",
    target="",
    context="",
    documents=None,
):
    conversation = conversation or []
    documents = documents or []

    mode_prompt = get_mode_prompt(mode)

    prompt_parts = [
        BASE_SYSTEM_PROMPT,
        "",
        mode_prompt,
        "",
        "==================================================",
        "CONTEXTO DE LA SESIÓN",
        "==================================================",
        f"Modo actual: {mode}",
    ]

    if target.strip():
        prompt_parts.append(
            f"Objetivo autorizado declarado: {target.strip()}"
        )

    if context.strip():
        prompt_parts.append(
            f"Contexto adicional: {context.strip()}"
        )

    if documents:
        prompt_parts.extend(
            [
                "",
                "==================================================",
                "DOCUMENTACIÓN LOCAL RELEVANTE",
                "==================================================",
            ]
        )

        for document in documents:
            prompt_parts.append(
                f"\n--- {document['name']} ---\n"
                f"{document['content']}"
            )

    prompt_parts.extend(
        [
            "",
            "==================================================",
            "HISTORIAL RECIENTE",
            "==================================================",
        ]
    )

    for message in conversation[-12:]:
        if not isinstance(message, dict):
            continue

        role = message.get("role", "")
        content = message.get("content", "")

        if role not in {"user", "assistant"}:
            continue

        if not content:
            continue

        label = (
            "USUARIO"
            if role == "user"
            else "ASISTENTE"
        )

        prompt_parts.append(
            f"{label}: {content}"
        )

    prompt_parts.extend(
        [
            "",
            "==================================================",
            "NUEVA PETICIÓN",
            "==================================================",
            question.strip(),
            "",
            "Responde de forma técnica, clara y estructurada.",
        ]
    )

    return "\n".join(prompt_parts)


def ask_ollama(prompt: str):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.15,
            "top_p": 0.9,
            "num_ctx": 8192,
        },
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=(15, 900),
        )

    except requests.exceptions.ConnectTimeout:
        raise RuntimeError(
            "Timeout conectando con Ollama."
        )

    except requests.exceptions.ReadTimeout:
        raise RuntimeError(
            "Ollama está tardando demasiado en responder."
        )

    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            f"No se puede conectar con Ollama: {exc}"
        )

    except requests.RequestException as exc:
        raise RuntimeError(
            f"Error comunicando con Ollama: {exc}"
        )

    if response.status_code != 200:
        detail = response.text[:1500]

        raise RuntimeError(
            f"Ollama devolvió HTTP {response.status_code}: "
            f"{detail}"
        )

    try:
        data = response.json()
    except Exception:
        raise RuntimeError(
            "Ollama devolvió una respuesta que no es JSON."
        )

    answer = data.get("response")

    if not answer:
        raise RuntimeError(
            "Ollama no devolvió texto en la respuesta."
        )

    return answer.strip()


# ============================================================
# ROUTES
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(HTML)


@app.get("/health")
async def health():
    try:
        response = requests.get(
            "http://ollama:11434/api/tags",
            timeout=10,
        )

        ollama_ok = response.status_code == 200

    except Exception:
        ollama_ok = False

    documents = [
        p.name
        for p in DOCUMENTS_DIR.iterdir()
        if p.is_file()
    ]

    return {
        "status": "ok",
        "app": APP_NAME,
        "model": MODEL,
        "ollama": ollama_ok,
        "documents": len(documents),
    }


@app.post("/ask")
async def ask(request: AskRequest):
    question = request.question.strip()

    if not question:
        return JSONResponse(
            status_code=400,
            content={
                "error": "La pregunta está vacía."
            },
        )

    documents = search_documents(question)

    try:
        prompt = build_prompt(
            question=question,
            conversation=request.conversation,
            mode=request.mode or "blue",
            target=request.target or "",
            context=request.context or "",
            documents=documents,
        )

        answer = ask_ollama(prompt)

        return {
            "answer": answer,
            "mode": request.mode or "blue",
            "model": MODEL,
            "documents": [
                {
                    "name": d["name"],
                    "score": d["score"],
                }
                for d in documents
            ],
        }

    except RuntimeError as exc:
        return JSONResponse(
            status_code=502,
            content={
                "error": str(exc)
            },
        )

    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Error interno: {exc}"
            },
        )


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
):
    filename = Path(file.filename or "").name

    if not filename:
        return JSONResponse(
            status_code=400,
            content={
                "error": "Nombre de archivo inválido."
            },
        )

    extension = Path(filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        return JSONResponse(
            status_code=400,
            content={
                "error": (
                    "Formato no soportado. "
                    "Usa PDF, TXT, MD, LOG, CSV, JSON, XML, "
                    "YAML o YML."
                )
            },
        )

    destination = DOCUMENTS_DIR / filename

    content = await file.read()

    if len(content) > 25 * 1024 * 1024:
        return JSONResponse(
            status_code=413,
            content={
                "error": "El archivo supera 25 MB."
            },
        )

    destination.write_bytes(content)

    return {
        "success": True,
        "filename": filename,
        "size": len(content),
    }


@app.get("/documents")
async def documents():
    items = []

    for path in sorted(DOCUMENTS_DIR.iterdir()):
        if not path.is_file():
            continue

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        try:
            size = path.stat().st_size
        except Exception:
            size = 0

        items.append(
            {
                "name": path.name,
                "size": size,
            }
        )

    return {
        "documents": items
    }


# ============================================================
# FRONTEND
# ============================================================

HTML = r"""
<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>CyberLocal AI</title>

<style>

:root {
    --bg: #070b12;
    --panel: #0d131d;
    --panel2: #111a26;
    --border: #1d2a3a;
    --text: #e7edf5;
    --muted: #8290a3;
    --green: #35e59a;
    --red: #ff5577;
    --blue: #4da3ff;
    --purple: #b778ff;
    --yellow: #ffd166;
}

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    height: 100vh;
    overflow: hidden;
    background: var(--bg);
    color: var(--text);
    font-family:
        Inter,
        system-ui,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;
}

button,
input,
textarea {
    font: inherit;
}

button {
    cursor: pointer;
}

.app {
    display: flex;
    height: 100vh;
}

/* =========================================================
   SIDEBAR
   ========================================================= */

.sidebar {
    width: 310px;
    min-width: 310px;
    border-right: 1px solid var(--border);
    background: #090e16;
    display: flex;
    flex-direction: column;
}

.brand {
    padding: 20px;
    border-bottom: 1px solid var(--border);
}

.brand-title {
    font-size: 20px;
    font-weight: 800;
    letter-spacing: .3px;
}

.brand-title span {
    color: var(--green);
}

.brand-subtitle {
    color: var(--muted);
    font-size: 12px;
    margin-top: 5px;
}

.new-chat {
    margin: 14px;
    padding: 11px 13px;
    border: 1px solid #284033;
    background: #0c1714;
    color: var(--green);
    border-radius: 9px;
    font-weight: 700;
}

.new-chat:hover {
    background: #10231d;
}

.sidebar-search {
    margin: 0 14px 10px;
}

.sidebar-search input {
    width: 100%;
    padding: 9px 10px;
    background: #0d141e;
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 8px;
    outline: none;
}

.chat-list {
    flex: 1;
    overflow-y: auto;
    padding: 0 9px 10px;
}

.chat-item {
    padding: 11px 10px;
    margin-bottom: 4px;
    border-radius: 8px;
    color: #c6d0dc;
    cursor: pointer;
    display: flex;
    justify-content: space-between;
    gap: 8px;
}

.chat-item:hover {
    background: #111a25;
}

.chat-item.active {
    background: #142019;
    border: 1px solid #254334;
}

.chat-title {
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    flex: 1;
}

.chat-actions {
    display: none;
    gap: 4px;
}

.chat-item:hover .chat-actions {
    display: flex;
}

.chat-actions button {
    border: 0;
    background: transparent;
    color: var(--muted);
    padding: 2px 4px;
}

.sidebar-bottom {
    border-top: 1px solid var(--border);
    padding: 12px;
}

.documents-title {
    color: var(--muted);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 7px;
}

.document-list {
    max-height: 130px;
    overflow-y: auto;
}

.document {
    font-size: 12px;
    color: #aebaca;
    padding: 4px 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.upload-label {
    display: block;
    text-align: center;
    margin-top: 8px;
    padding: 8px;
    border: 1px dashed #2a394c;
    border-radius: 8px;
    color: var(--muted);
    cursor: pointer;
}

.upload-label:hover {
    color: var(--text);
    border-color: #42607e;
}

#fileInput {
    display: none;
}

/* =========================================================
   MAIN
   ========================================================= */

.main {
    flex: 1;
    min-width: 0;
    display: flex;
    flex-direction: column;
}

.topbar {
    min-height: 67px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 18px;
    background: #090e16;
}

.top-left {
    display: flex;
    align-items: center;
    gap: 12px;
}

.mode-switch {
    display: flex;
    background: #0d141e;
    border: 1px solid var(--border);
    border-radius: 9px;
    padding: 3px;
    gap: 3px;
}

.mode-btn {
    border: 0;
    background: transparent;
    color: var(--muted);
    padding: 8px 13px;
    border-radius: 7px;
    font-weight: 700;
    font-size: 12px;
}

.mode-btn.active.blue {
    background: #102238;
    color: var(--blue);
}

.mode-btn.active.red {
    background: #2a1119;
    color: var(--red);
}

.mode-btn.active.purple {
    background: #21162e;
    color: var(--purple);
}

.status {
    font-size: 12px;
    color: var(--muted);
}

.status-dot {
    display: inline-block;
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--green);
    margin-right: 5px;
}

.toolbar {
    display: flex;
    gap: 6px;
}

.tool-button {
    border: 1px solid var(--border);
    background: #0d141e;
    color: #aebaca;
    border-radius: 7px;
    padding: 7px 9px;
}

.tool-button:hover {
    color: var(--text);
    background: #121c28;
}

/* =========================================================
   CONTEXT
   ========================================================= */

.context-bar {
    padding: 9px 18px;
    border-bottom: 1px solid var(--border);
    background: #0a1018;
    display: flex;
    gap: 8px;
}

.context-input {
    flex: 1;
    min-width: 0;
    padding: 8px 10px;
    border: 1px solid var(--border);
    background: #0d141e;
    color: var(--text);
    border-radius: 7px;
    outline: none;
    font-size: 12px;
}

.context-input:focus {
    border-color: #38546e;
}

/* =========================================================
   QUICK ACTIONS
   ========================================================= */

.quick-actions {
    display: flex;
    gap: 7px;
    padding: 9px 18px;
    overflow-x: auto;
    border-bottom: 1px solid #111b27;
}

.quick {
    flex-shrink: 0;
    border: 1px solid var(--border);
    background: #0d141e;
    color: #aab6c6;
    border-radius: 7px;
    padding: 7px 10px;
    font-size: 11px;
}

.quick:hover {
    color: var(--text);
    border-color: #38506a;
}

/* =========================================================
   CHAT
   ========================================================= */

.chat {
    flex: 1;
    overflow-y: auto;
    padding: 24px max(18px, calc((100% - 950px) / 2));
}

.welcome {
    max-width: 760px;
    margin: 60px auto;
    text-align: center;
}

.welcome h1 {
    font-size: 30px;
    margin: 0 0 10px;
}

.welcome h1 span {
    color: var(--green);
}

.welcome p {
    color: var(--muted);
    line-height: 1.6;
}

.message {
    display: flex;
    margin-bottom: 18px;
    gap: 11px;
}

.message.user {
    justify-content: flex-end;
}

.message-bubble {
    max-width: 82%;
    padding: 12px 14px;
    border-radius: 10px;
    line-height: 1.55;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
}

.message.assistant .message-bubble {
    background: #0e1722;
    border: 1px solid var(--border);
}

.message.user .message-bubble {
    background: #13251e;
    border: 1px solid #254334;
}

.message-meta {
    font-size: 10px;
    color: var(--muted);
    margin-bottom: 5px;
}

.typing {
    color: var(--muted);
    font-size: 13px;
    padding: 10px 0;
}

/* =========================================================
   INPUT
   ========================================================= */

.composer-wrap {
    border-top: 1px solid var(--border);
    background: #090e16;
    padding: 13px 18px 16px;
}

.composer {
    max-width: 950px;
    margin: auto;
    position: relative;
}

textarea {
    width: 100%;
    min-height: 78px;
    max-height: 220px;
    resize: vertical;
    padding: 12px 54px 12px 13px;
    border-radius: 10px;
    border: 1px solid var(--border);
    background: #0d141e;
    color: var(--text);
    outline: none;
}

textarea:focus {
    border-color: #38546e;
}

.send {
    position: absolute;
    right: 8px;
    bottom: 8px;
    width: 36px;
    height: 36px;
    border: 0;
    border-radius: 8px;
    background: var(--green);
    color: #06110c;
    font-weight: 900;
}

.send:disabled {
    opacity: .45;
}

.composer-hint {
    max-width: 950px;
    margin: 6px auto 0;
    color: #586779;
    font-size: 10px;
    text-align: center;
}

/* =========================================================
   MODAL
   ========================================================= */

.modal {
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,.65);
    display: none;
    align-items: center;
    justify-content: center;
    z-index: 50;
}

.modal.show {
    display: flex;
}

.modal-card {
    width: min(600px, calc(100% - 30px));
    background: #0d141e;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 18px;
}

.modal-card h3 {
    margin-top: 0;
}

.modal-card textarea {
    min-height: 260px;
    margin-top: 10px;
}

.modal-buttons {
    display: flex;
    justify-content: flex-end;
    gap: 8px;
    margin-top: 10px;
}

.modal-buttons button {
    border: 1px solid var(--border);
    background: #121c28;
    color: var(--text);
    border-radius: 7px;
    padding: 8px 13px;
}

@media (max-width: 850px) {

    .sidebar {
        width: 235px;
        min-width: 235px;
    }

    .context-bar {
        flex-direction: column;
    }

    .message-bubble {
        max-width: 92%;
    }
}

@media (max-width: 650px) {

    .sidebar {
        display: none;
    }

    .topbar {
        padding: 8px;
    }

    .status {
        display: none;
    }

    .mode-btn {
        padding: 7px 9px;
    }

    .toolbar button:nth-child(1) {
        display: none;
    }

    .chat {
        padding: 15px 10px;
    }

    .composer-wrap {
        padding: 9px;
    }
}

</style>
</head>

<body>

<div class="app">

    <!-- ====================================================
         SIDEBAR
         ==================================================== -->

    <aside class="sidebar">

        <div class="brand">
            <div class="brand-title">
                <span>Cyber</span>Local AI
            </div>

            <div class="brand-subtitle">
                Local Cybersecurity Assistant
            </div>
        </div>

        <button class="new-chat" onclick="newChat()">
            + Nueva conversación
        </button>

        <div class="sidebar-search">
            <input
                id="chatSearch"
                placeholder="Buscar conversaciones..."
                oninput="renderChatList()"
            >
        </div>

        <div
            class="chat-list"
            id="chatList"
        ></div>

        <div class="sidebar-bottom">

            <div class="documents-title">
                Documentos locales
            </div>

            <div
                class="document-list"
                id="documentList"
            ></div>

            <label class="upload-label">
                + Subir documento
                <input
                    id="fileInput"
                    type="file"
                    onchange="uploadDocument()"
                >
            </label>

        </div>

    </aside>


    <!-- ====================================================
         MAIN
         ==================================================== -->

    <main class="main">

        <header class="topbar">

            <div class="top-left">

                <div class="mode-switch">

                    <button
                        class="mode-btn blue active"
                        data-mode="blue"
                        onclick="setMode('blue')"
                    >
                        🟦 Blue Team
                    </button>

                    <button
                        class="mode-btn red"
                        data-mode="red"
                        onclick="setMode('red')"
                    >
                        🟥 Red Team
                    </button>

                    <button
                        class="mode-btn purple"
                        data-mode="purple"
                        onclick="setMode('purple')"
                    >
                        🟪 Purple
                    </button>

                </div>

                <div class="status">
                    <span
                        class="status-dot"
                        id="statusDot"
                    ></span>
                    <span id="statusText">
                        Comprobando...
                    </span>
                </div>

            </div>

            <div class="toolbar">

                <button
                    class="tool-button"
                    onclick="renameCurrentChat()"
                    title="Renombrar"
                >
                    ✏️
                </button>

                <button
                    class="tool-button"
                    onclick="exportChat()"
                    title="Exportar"
                >
                    📥
                </button>

                <button
                    class="tool-button"
                    onclick="clearCurrentChat()"
                    title="Limpiar"
                >
                    🧹
                </button>

            </div>

        </header>


        <!-- CONTEXT -->

        <div class="context-bar">

            <input
                id="targetInput"
                class="context-input"
                placeholder="Objetivo / entorno autorizado (ej: laboratorio AD, staging, CTF...)"
            >

            <input
                id="contextInput"
                class="context-input"
                placeholder="Contexto adicional: tecnologías, IPs, logs, alcance..."
            >

        </div>


        <!-- QUICK ACTIONS -->

        <div
            class="quick-actions"
            id="quickActions"
        ></div>


        <!-- CHAT -->

        <section
            class="chat"
            id="chat"
        ></section>


        <!-- COMPOSER -->

        <div class="composer-wrap">

            <div class="composer">

                <textarea
                    id="question"
                    placeholder="Pregunta a CyberLocal AI..."
                    onkeydown="handleKey(event)"
                ></textarea>

                <button
                    class="send"
                    id="sendButton"
                    onclick="sendMessage()"
                >
                    ➤
                </button>

            </div>

            <div class="composer-hint">
                Enter para enviar · Shift+Enter para nueva línea
            </div>

        </div>

    </main>

</div>


<!-- EXPORT MODAL -->

<div
    class="modal"
    id="exportModal"
    onclick="closeExport(event)"
>

    <div
        class="modal-card"
        onclick="event.stopPropagation()"
    >

        <h3>Exportar conversación</h3>

        <textarea
            id="exportText"
            readonly
        ></textarea>

        <div class="modal-buttons">

            <button onclick="copyExport()">
                Copiar
            </button>

            <button onclick="downloadExport()">
                Descargar
            </button>

            <button onclick="closeExport()">
                Cerrar
            </button>

        </div>

    </div>

</div>


<script>

/* =========================================================
   STATE
   ========================================================= */

const STORAGE_KEY = "cyberlocal_chats_v3";

let chats = [];

let currentChatId = null;

let currentMode = "blue";

let sending = false;


/* =========================================================
   MODE CONFIG
   ========================================================= */

const MODE_CONFIG = {

    blue: {
        label: "Blue Team",
        quick: [
            ["🔎 Threat Hunting", "Haz un plan de threat hunting para este entorno."],
            ["🚨 Incident Response", "Analiza este escenario como incidente y dame un plan de respuesta."],
            ["🧬 IOC Analysis", "Analiza estos indicadores y dime qué evidencias buscar."],
            ["📊 SIEM", "Propón queries de detección para este escenario."],
            ["🛡️ Hardening", "Haz un checklist de hardening para este sistema."],
            ["🕵️ Forensics", "Propón una metodología de análisis forense."]
        ]
    },

    red: {
        label: "Red Team",
        quick: [
            ["🎯 Attack Surface", "Ayúdame a mapear la superficie de ataque del objetivo autorizado."],
            ["🔍 Recon", "Diseña una metodología de reconocimiento autorizado."],
            ["🌐 Web Security", "Analiza este escenario desde la perspectiva de seguridad web."],
            ["🔐 AD Assessment", "Diseña una evaluación de seguridad de Active Directory."],
            ["⬆️ PrivEsc", "Analiza posibles rutas de escalada de privilegios en este laboratorio."],
            ["↔️ Lateral Movement", "Analiza posibles rutas de movimiento lateral y qué evidencias generarían."],
            ["🎭 Adversary Emulation", "Diseña un ejercicio de emulación de adversario controlado."]
        ]
    },

    purple: {
        label: "Purple Team",
        quick: [
            ["⚔️ ATT&CK", "Relaciona este escenario con MITRE ATT&CK y sus posibles detecciones."],
            ["🎯 Attack → Detect", "Diseña una técnica ofensiva autorizada y las detecciones correspondientes."],
            ["🧪 Validation", "Diseña una prueba controlada para validar las defensas."],
            ["📡 Telemetry", "¿Qué telemetría debería producir este ataque y dónde buscarla?"],
            ["🛡️ Improve", "Analiza cómo mejorar simultáneamente ataque y defensa."]
        ]
    }

};


/* =========================================================
   STORAGE
   ========================================================= */

function saveChats() {

    localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify(chats)
    );
}


function loadChats() {

    try {

        const raw = localStorage.getItem(
            STORAGE_KEY
        );

        chats = raw
            ? JSON.parse(raw)
            : [];

    } catch (error) {

        chats = [];

    }

    if (!Array.isArray(chats)) {
        chats = [];
    }

    if (!chats.length) {
        newChat();
        return;
    }

    currentChatId =
        localStorage.getItem(
            "cyberlocal_current_chat"
        ) || chats[0].id;

    if (!findCurrentChat()) {
        currentChatId = chats[0].id;
    }

    renderChatList();
    renderCurrentChat();
}


/* =========================================================
   CHAT MANAGEMENT
   ========================================================= */

function generateId() {

    return (
        Date.now().toString(36) +
        Math.random()
            .toString(36)
            .substring(2, 8)
    );
}


function createChat(title = "Nueva conversación") {

    return {
        id: generateId(),
        title: title,
        createdAt: Date.now(),
        updatedAt: Date.now(),
        mode: currentMode,
        target: "",
        context: "",
        messages: []
    };
}


function findCurrentChat() {

    return chats.find(
        chat => chat.id === currentChatId
    );
}


function newChat() {

    const chat = createChat();

    chats.unshift(chat);

    currentChatId = chat.id;

    localStorage.setItem(
        "cyberlocal_current_chat",
        currentChatId
    );

    saveChats();

    renderChatList();
    renderCurrentChat();
}


function selectChat(id) {

    currentChatId = id;

    localStorage.setItem(
        "cyberlocal_current_chat",
        currentChatId
    );

    const chat = findCurrentChat();

    if (chat) {

        currentMode =
            chat.mode || "blue";

        document.getElementById(
            "targetInput"
        ).value = chat.target || "";

        document.getElementById(
            "contextInput"
        ).value = chat.context || "";

        updateModeUI();

    }

    renderChatList();
    renderCurrentChat();
}


function deleteChat(id, event) {

    if (event) {
        event.stopPropagation();
    }

    const chat = chats.find(
        item => item.id === id
    );

    if (!chat) {
        return;
    }

    if (
        !confirm(
            `¿Eliminar "${chat.title}"?`
        )
    ) {
        return;
    }

    chats = chats.filter(
        item => item.id !== id
    );

    if (!chats.length) {

        const fresh = createChat();

        chats.push(fresh);

        currentChatId = fresh.id;

    } else if (currentChatId === id) {

        currentChatId =
            chats[0].id;

    }

    saveChats();

    localStorage.setItem(
        "cyberlocal_current_chat",
        currentChatId
    );

    renderChatList();
    renderCurrentChat();
}


function renameChat(id, event) {

    if (event) {
        event.stopPropagation();
    }

    const chat = chats.find(
        item => item.id === id
    );

    if (!chat) {
        return;
    }

    const title = prompt(
        "Nuevo nombre:",
        chat.title
    );

    if (
        title === null ||
        !title.trim()
    ) {
        return;
    }

    chat.title =
        title.trim().substring(0, 100);

    chat.updatedAt = Date.now();

    saveChats();

    renderChatList();
}


function renameCurrentChat() {

    if (!currentChatId) {
        return;
    }

    renameChat(currentChatId);
}


function clearCurrentChat() {

    const chat = findCurrentChat();

    if (!chat) {
        return;
    }

    if (
        !confirm(
            "¿Limpiar todos los mensajes de esta conversación?"
        )
    ) {
        return;
    }

    chat.messages = [];
    chat.updatedAt = Date.now();

    saveChats();

    renderCurrentChat();
}


/* =========================================================
   SIDEBAR
   ========================================================= */

function renderChatList() {

    const container =
        document.getElementById("chatList");

    const search =
        (
            document.getElementById(
                "chatSearch"
            ).value || ""
        )
        .toLowerCase()
        .trim();

    const filtered = chats.filter(
        chat => {

            if (!search) {
                return true;
            }

            const haystack = (
                chat.title +
                " " +
                chat.messages
                    .map(m => m.content)
                    .join(" ")
            ).toLowerCase();

            return haystack.includes(search);
        }
    );

    container.innerHTML = "";

    filtered
        .sort(
            (a, b) =>
                b.updatedAt - a.updatedAt
        )
        .forEach(chat => {

            const item =
                document.createElement("div");

            item.className =
                "chat-item" +
                (
                    chat.id === currentChatId
                        ? " active"
                        : ""
                );

            item.onclick = () =>
                selectChat(chat.id);

            const title =
                document.createElement("div");

            title.className =
                "chat-title";

            title.textContent =
                chat.title ||
                "Nueva conversación";

            const actions =
                document.createElement("div");

            actions.className =
                "chat-actions";

            const rename =
                document.createElement("button");

            rename.textContent = "✏️";

            rename.title = "Renombrar";

            rename.onclick = event =>
                renameChat(
                    chat.id,
                    event
                );

            const remove =
                document.createElement("button");

            remove.textContent = "×";

            remove.title = "Eliminar";

            remove.onclick = event =>
                deleteChat(
                    chat.id,
                    event
                );

            actions.appendChild(rename);
            actions.appendChild(remove);

            item.appendChild(title);
            item.appendChild(actions);

            container.appendChild(item);

        });
}


/* =========================================================
   MODE
   ========================================================= */

function setMode(mode) {

    currentMode = mode;

    const chat = findCurrentChat();

    if (chat) {
        chat.mode = mode;
        chat.updatedAt = Date.now();
        saveChats();
    }

    updateModeUI();
    renderQuickActions();
}


function updateModeUI() {

    document
        .querySelectorAll(".mode-btn")
        .forEach(button => {

            button.classList.remove(
                "active"
            );

            if (
                button.dataset.mode ===
                currentMode
            ) {
                button.classList.add(
                    "active"
                );
            }

        });
}


function renderQuickActions() {

    const container =
        document.getElementById(
            "quickActions"
        );

    container.innerHTML = "";

    const config =
        MODE_CONFIG[currentMode];

    config.quick.forEach(
        ([label, prompt]) => {

            const button =
                document.createElement("button");

            button.className = "quick";

            button.textContent = label;

            button.onclick = () => {

                document.getElementById(
                    "question"
                ).value = prompt;

                document.getElementById(
                    "question"
                ).focus();

            };

            container.appendChild(button);

        }
    );
}


/* =========================================================
   CHAT RENDER
   ========================================================= */

function escapeHtml(text) {

    return String(text)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}


function renderCurrentChat() {

    const container =
        document.getElementById("chat");

    const chat = findCurrentChat();

    if (!chat) {
        return;
    }

    container.innerHTML = "";

    if (!chat.messages.length) {

        const welcome =
            document.createElement("div");

        welcome.className = "welcome";

        welcome.innerHTML = `
            <h1>
                <span>Cyber</span>Local AI
            </h1>

            <p>
                Asistente local de ciberseguridad.
                Selecciona un modo y empieza a trabajar.
            </p>

            <p>
                <strong>${MODE_CONFIG[currentMode].label}</strong>
                activo.
            </p>
        `;

        container.appendChild(welcome);

        return;
    }

    chat.messages.forEach(
        message => {

            const wrapper =
                document.createElement("div");

            wrapper.className =
                "message " +
                (
                    message.role === "user"
                        ? "user"
                        : "assistant"
                );

            const bubble =
                document.createElement("div");

            bubble.className =
                "message-bubble";

            bubble.textContent =
                message.content;

            wrapper.appendChild(
                bubble
            );

            container.appendChild(
                wrapper
            );

        }
    );

    container.scrollTop =
        container.scrollHeight;
}


/* =========================================================
   SEND
   ========================================================= */

async function sendMessage() {

    if (sending) {
        return;
    }

    const input =
        document.getElementById(
            "question"
        );

    const question =
        input.value.trim();

    if (!question) {
        return;
    }

    const chat = findCurrentChat();

    if (!chat) {
        return;
    }

    const target =
        document.getElementById(
            "targetInput"
        ).value.trim();

    const context =
        document.getElementById(
            "contextInput"
        ).value.trim();

    chat.target = target;
    chat.context = context;
    chat.mode = currentMode;

    if (
        chat.messages.length === 0
    ) {

        chat.title =
            question
                .replace(/\s+/g, " ")
                .substring(0, 55);

    }

    chat.messages.push({
        role: "user",
        content: question
    });

    chat.updatedAt = Date.now();

    input.value = "";

    saveChats();

    renderChatList();
    renderCurrentChat();

    sending = true;

    const sendButton =
        document.getElementById(
            "sendButton"
        );

    sendButton.disabled = true;

    const typing =
        document.createElement("div");

    typing.className =
        "typing";

    typing.id =
        "typingIndicator";

    typing.textContent =
        "CyberLocal AI está analizando...";

    document
        .getElementById("chat")
        .appendChild(typing);

    document
        .getElementById("chat")
        .scrollTop =
        document
            .getElementById("chat")
            .scrollHeight;

    try {

        const response =
            await fetch(
                "/ask",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({
                        question: question,
                        conversation:
                            chat.messages.slice(
                                -13,
                                -1
                            ),
                        mode: currentMode,
                        target: target,
                        context: context
                    })
                }
            );

        const raw =
            await response.text();

        let data;

        try {

            data =
                JSON.parse(raw);

        } catch (error) {

            throw new Error(
                "El servidor devolvió una respuesta no válida.\n\n" +
                "HTTP: " +
                response.status +
                "\n\n" +
                raw.substring(0, 1200)
            );

        }

        if (!response.ok) {

            throw new Error(
                data.detail ||
                data.error ||
                "Error HTTP " +
                response.status
            );

        }

        const answer =
            data.answer ||
            "No se recibió respuesta del modelo.";

        chat.messages.push({
            role: "assistant",
            content: answer
        });

        chat.updatedAt = Date.now();

        saveChats();

        renderChatList();
        renderCurrentChat();

    } catch (error) {

        chat.messages.push({
            role: "assistant",
            content:
                "❌ No se pudo obtener respuesta del modelo.\n\n" +
                error.message
        });

        chat.updatedAt = Date.now();

        saveChats();

        renderCurrentChat();

    } finally {

        sending = false;

        sendButton.disabled = false;

        const indicator =
            document.getElementById(
                "typingIndicator"
            );

        if (indicator) {
            indicator.remove();
        }

    }
}


/* =========================================================
   KEYBOARD
   ========================================================= */

function handleKey(event) {

    if (
        event.key === "Enter" &&
        !event.shiftKey
    ) {

        event.preventDefault();

        sendMessage();

    }

}


/* =========================================================
   EXPORT
   ========================================================= */

function formatChatForExport() {

    const chat = findCurrentChat();

    if (!chat) {
        return "";
    }

    let output =
        "# CyberLocal AI\n\n";

    output +=
        `## ${chat.title}\n\n`;

    output +=
        `Modo: ${chat.mode}\n`;

    if (chat.target) {
        output +=
            `Objetivo: ${chat.target}\n`;
    }

    if (chat.context) {
        output +=
            `Contexto: ${chat.context}\n`;
    }

    output += "\n---\n\n";

    chat.messages.forEach(
        message => {

            output +=
                message.role === "user"
                    ? "## Usuario\n\n"
                    : "## CyberLocal AI\n\n";

            output +=
                message.content +
                "\n\n";

        }
    );

    return output;
}


function exportChat() {

    const text =
        formatChatForExport();

    document.getElementById(
        "exportText"
    ).value = text;

    document.getElementById(
        "exportModal"
    ).classList.add("show");
}


function closeExport(event) {

    if (
        !event ||
        event.target.id ===
        "exportModal"
    ) {

        document.getElementById(
            "exportModal"
        ).classList.remove("show");

    }
}


async function copyExport() {

    const text =
        document.getElementById(
            "exportText"
        ).value;

    try {

        await navigator.clipboard.writeText(
            text
        );

    } catch {

        document.getElementById(
            "exportText"
        ).select();

        document.execCommand("copy");

    }
}


function downloadExport() {

    const text =
        formatChatForExport();

    const blob =
        new Blob(
            [text],
            {
                type:
                    "text/markdown;charset=utf-8"
            }
        );

    const url =
        URL.createObjectURL(blob);

    const a =
        document.createElement("a");

    a.href = url;

    a.download =
        (
            findCurrentChat()?.title ||
            "cyberlocal-chat"
        )
        .replace(
            /[^a-z0-9áéíóúüñ_-]+/gi,
            "_"
        )
        + ".md";

    document.body.appendChild(a);

    a.click();

    a.remove();

    URL.revokeObjectURL(url);
}


/* =========================================================
   DOCUMENTS
   ========================================================= */

async function loadDocuments() {

    try {

        const response =
            await fetch("/documents");

        const data =
            await response.json();

        const container =
            document.getElementById(
                "documentList"
            );

        container.innerHTML = "";

        (data.documents || [])
            .forEach(documentItem => {

                const element =
                    document.createElement(
                        "div"
                    );

                element.className =
                    "document";

                element.title =
                    documentItem.name;

                element.textContent =
                    "📄 " +
                    documentItem.name;

                container.appendChild(
                    element
                );

            });

    } catch {

        document.getElementById(
            "documentList"
        ).innerHTML =
            "<div class='document'>Sin conexión</div>";

    }
}


async function uploadDocument() {

    const input =
        document.getElementById(
            "fileInput"
        );

    const file =
        input.files[0];

    if (!file) {
        return;
    }

    const form =
        new FormData();

    form.append(
        "file",
        file
    );

    try {

        const response =
            await fetch(
                "/upload",
                {
                    method: "POST",
                    body: form
                }
            );

        const data =
            await response.json();

        if (!response.ok) {

            alert(
                data.error ||
                "Error subiendo documento."
            );

            return;
        }

        await loadDocuments();

    } catch (error) {

        alert(
            "Error subiendo documento: " +
            error.message
        );

    } finally {

        input.value = "";

    }
}


/* =========================================================
   HEALTH
   ========================================================= */

async function checkHealth() {

    try {

        const response =
            await fetch("/health");

        const data =
            await response.json();

        const dot =
            document.getElementById(
                "statusDot"
            );

        const text =
            document.getElementById(
                "statusText"
            );

        if (
            response.ok &&
            data.ollama
        ) {

            dot.style.background =
                "var(--green)";

            text.textContent =
                "Ollama conectado";

        } else {

            dot.style.background =
                "var(--yellow)";

            text.textContent =
                "Ollama no disponible";

        }

    } catch {

        document.getElementById(
            "statusDot"
        ).style.background =
            "var(--red)";

        document.getElementById(
            "statusText"
        ).textContent =
            "Servidor no disponible";

    }
}


/* =========================================================
   CONTEXT AUTOSAVE
   ========================================================= */

function setupContextAutosave() {

    [
        "targetInput",
        "contextInput"
    ].forEach(id => {

        document
            .getElementById(id)
            .addEventListener(
                "input",
                () => {

                    const chat =
                        findCurrentChat();

                    if (!chat) {
                        return;
                    }

                    chat.target =
                        document.getElementById(
                            "targetInput"
                        ).value;

                    chat.context =
                        document.getElementById(
                            "contextInput"
                        ).value;

                    chat.updatedAt =
                        Date.now();

                    saveChats();

                }
            );

    });
}


/* =========================================================
   INIT
   ========================================================= */

function init() {

    loadChats();

    updateModeUI();

    renderQuickActions();

    loadDocuments();

    checkHealth();

    setupContextAutosave();

    setInterval(
        checkHealth,
        30000
    );

}


init();

</script>

</body>
</html>
"""
