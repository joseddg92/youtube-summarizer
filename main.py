#!/usr/bin/env python3
"""
Bot de Telegram que resume vídeos de YouTube a partir de sus subtítulos (vía yt-dlp) usando un
LLM compatible con OpenAI (Inference API de Hetzner por defecto).

Cualquier usuario, grupo o canal puede usarlo:
    /video <url>          resume un vídeo (pregunta cómo resumirlo)
    /suscribirse <url>    avisa con un resumen de cada vídeo nuevo de un canal de YouTube
    /suscripciones        lista las suscripciones del chat; permite editar el prompt o borrarlas
    /ayuda                ayuda

Todas las acciones se notifican al chat de administración (TELEGRAM_CHAT_ID).

Uso:
    python main.py                # bot: escucha Telegram y comprueba los canales cada POLL_INTERVAL_MINUTES
    python main.py --once         # una sola pasada por los canales suscritos (cron; no escucha Telegram)
    python main.py --get-chat-id  # ayuda para obtener TELEGRAM_CHAT_ID
    python main.py --video URL    # resume un vídeo y lo envía al chat de administración
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import secrets
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
import yt_dlp
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

log = logging.getLogger("yt-summarizer")

# --------------------------------------------------------------------------- config

def env(name: str, default: str | None = None, required: bool = False) -> str:
    value = os.environ.get(name, default)
    if required and not value:
        sys.exit(f"Falta la variable de entorno {name} (revisa tu .env)")
    return value or ""


CONFIG = {
    "hetzner_api_key": env("HETZNER_INFERENCE_API_KEY"),
    "hetzner_base_url": env("HETZNER_INFERENCE_BASE_URL", "https://inference.hetzner.com/api/v1"),
    "hetzner_model": env("HETZNER_INFERENCE_MODEL", "Qwen/Qwen3.6-35B-A3B-FP8"),
    "subtitle_langs": [l.strip() for l in env("SUBTITLE_LANGUAGES", "en,es").split(",") if l.strip()],
    "check_latest_n": int(env("CHECK_LATEST_N", "10")),
    "max_per_run": int(env("MAX_VIDEOS_PER_RUN", "3")),
    "give_up_hours": float(env("GIVE_UP_AFTER_HOURS", "48")),
    "cookies_file": env("YTDLP_COOKIES_FILE"),
    "cookies_from_browser": env("YTDLP_COOKIES_FROM_BROWSER"),
    "telegram_token": env("TELEGRAM_BOT_API_KEY"),
    "admin_chat_id": env("TELEGRAM_CHAT_ID"),  # chat donde se notifican todas las acciones
    "summary_language": env("SUMMARY_LANGUAGE", "español"),
    "poll_minutes": float(env("POLL_INTERVAL_MINUTES", "30")),
    "state_file": Path(env("STATE_FILE", "state.json")),
    "subscriptions_file": Path(env("SUBSCRIPTIONS_FILE", "subscriptions.json")),
    "legacy_channels_file": Path(env("CHANNELS_FILE", "channels.json")),  # formato antiguo, se migra
    "workers": int(env("WORKERS", "3")),  # resúmenes en paralelo
}

# Límite de seguridad para la transcripción (~100k tokens; el modelo admite 262k)
MAX_TRANSCRIPT_CHARS = 400_000
TELEGRAM_MAX_LEN = 4000  # el límite real es 4096
TELEGRAM_LONG_POLL_SECONDS = 25
NETWORK_RETRIES = 3          # reintentos por llamada de red (Telegram, YouTube, Hetzner)
NETWORK_BACKOFF_SECONDS = 3  # espera inicial entre reintentos (se duplica en cada uno)
EXAMPLES_COUNT = 3                          # vídeos de ejemplo que se ofrecen al suscribirse
EXAMPLES_TIMEOUT = timedelta(minutes=30)    # sin respuesta: se borra la pregunta y se omite
PROMPT_TIMEOUT = timedelta(hours=24)        # preguntas de prompt sin contestar caducan

YOUTUBE_URL_RE = re.compile(
    r"(?:https?://)?(?:www\.|m\.)?"
    r"(?:youtube\.com/(?:watch\?(?:[^\s]*&)?v=|shorts/|live/|embed/)|youtu\.be/)"
    r"([\w-]{11})"
)

DEFAULT_INSTRUCTIONS = (
    "1. Un párrafo de 2-3 frases con la idea principal.\n"
    "2. 'Puntos clave:' seguido de 4-8 viñetas concretas (datos, cifras, argumentos, recomendaciones).\n"
    "3. 'Conclusión:' una frase con la postura o recomendación final del autor.\n"
    "Máximo ~300 palabras."
)

# --------------------------------------------------------------------------- errores y reintentos


def short_error(exc: BaseException) -> str:
    """Descripción de una línea de una excepción, sin el ruido de urllib3/requests."""
    # requests envuelve el error real varias veces; nos quedamos con la causa más profunda
    cause = exc
    seen = set()
    while id(cause) not in seen:
        seen.add(id(cause))
        nxt = cause.__cause__ or cause.__context__
        if nxt is None:
            break
        cause = nxt
    detail = (str(cause) or str(exc)).splitlines()[0] if (str(cause) or str(exc)) else ""
    name = type(exc).__name__
    return f"{name}: {detail}"[:300] if detail else name


def log_error(message: str, exc: BaseException) -> None:
    """Una línea en ERROR; el traceback completo solo con -v (DEBUG)."""
    log.error("%s: %s", message, short_error(exc))
    log.debug("Traceback:", exc_info=exc)


def with_retries(what: str, fn, attempts: int = NETWORK_RETRIES):
    """Ejecuta fn() reintentando con backoff exponencial."""
    delay = NETWORK_BACKOFF_SECONDS
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            if attempt == attempts:
                raise
            log.warning("%s falló (intento %d/%d): %s. Reintentando en %ds…",
                        what, attempt, attempts, short_error(exc), delay)
            time.sleep(delay)
            delay *= 2


# --------------------------------------------------------------------------- youtube


CHANNEL_TABS = ("videos", "streams")  # pestañas del canal que se pueden vigilar


def normalize_channel_url(channel: str, tab: str = "videos") -> str:
    """Convierte URL / @handle / UC... en la URL de una pestaña del canal (videos o streams)."""
    channel = channel.strip()
    if channel.startswith("http"):
        url = channel.rstrip("/")
    elif channel.startswith("@"):
        url = f"https://www.youtube.com/{channel}"
    elif channel.startswith("UC"):
        url = f"https://www.youtube.com/channel/{channel}"
    else:
        url = f"https://www.youtube.com/@{channel}"
    url = re.sub(r"/(videos|streams|shorts|live|featured)$", "", url)
    return f"{url}/{tab}"


def tab_from_url(ref: str) -> str | None:
    """Si la URL apunta a una pestaña concreta (/videos, /streams) la devuelve."""
    match = re.search(r"/(videos|streams)/?$", ref.strip())
    return match.group(1) if match else None


def channel_tabs(channel: dict) -> list[str]:
    return channel.get("tabs") or ["videos"]


def video_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def extract_video_ids(text: str) -> list[str]:
    """Devuelve los IDs de vídeo de YouTube que aparecen en un texto (sin duplicados, en orden)."""
    ids = []
    for match in YOUTUBE_URL_RE.finditer(text):
        vid = match.group(1)
        if vid not in ids:
            ids.append(vid)
    return ids


def ydl_base_opts() -> dict:
    opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noprogress": True,
    }
    if CONFIG["cookies_file"]:
        opts["cookiefile"] = CONFIG["cookies_file"]
    elif CONFIG["cookies_from_browser"]:
        # p. ej. "chrome", "firefox", "safari", "brave", "edge" (o "chrome:Profile 1")
        browser, _, profile = CONFIG["cookies_from_browser"].partition(":")
        opts["cookiesfrombrowser"] = (browser, profile or None)
    return opts


def fetch_channel(channel_url: str, limit: int) -> tuple[dict, list[dict]]:
    """
    Devuelve (info del canal, últimos `limit` vídeos más recientes primero) sin descargar nada.
    info contiene channel_id, channel (nombre) y uploader_id (@handle).
    """
    opts = ydl_base_opts() | {"extract_flat": "in_playlist", "playlistend": limit}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = with_retries("Listado del canal", lambda: ydl.extract_info(channel_url, download=False))
    videos = []
    for entry in info.get("entries") or []:
        vid = entry.get("id")
        if not vid:
            continue
        videos.append({"id": vid, "title": entry.get("title") or vid, "url": video_url(vid)})
    return info, videos


def resolve_channel(ref: str) -> dict:
    """Convierte una URL/@handle/ID (o la URL de un vídeo suyo) en un registro {id, name, handle, url, tabs}."""
    ids = extract_video_ids(ref)
    if ids:
        with yt_dlp.YoutubeDL(ydl_base_opts()) as ydl:
            info = with_retries("Info del vídeo", lambda: ydl.extract_info(video_url(ids[0]), download=False, process=False))
        ref = info.get("channel_url") or info.get("uploader_url") or ref
    tab = tab_from_url(ref) or "videos"
    info, _ = fetch_channel(normalize_channel_url(ref, tab), 1)
    channel_id = info.get("channel_id") or info.get("id")
    if not channel_id or not channel_id.startswith("UC"):
        raise ValueError(f"No parece un canal de YouTube: {ref}")
    handle = info.get("uploader_id") or ""
    return {
        "id": channel_id,
        "name": info.get("channel") or info.get("uploader") or handle or channel_id,
        "handle": handle if handle.startswith("@") else "",
        "url": f"https://www.youtube.com/{handle}" if handle.startswith("@") else f"https://www.youtube.com/channel/{channel_id}",
        "tabs": [tab],
    }


def latest_channel_videos(channel: dict, limit: int) -> list[dict]:
    """Últimos vídeos del canal (más recientes primero) juntando las pestañas que vigila."""
    videos: list[dict] = []
    for tab in channel_tabs(channel):
        _, tab_videos = fetch_channel(normalize_channel_url(channel["url"], tab), limit)
        known = {v["id"] for v in videos}
        videos += [v for v in tab_videos if v["id"] not in known]
    return videos


def parse_json3(raw: bytes) -> str:
    """Convierte el formato json3 de subtítulos de YouTube a texto plano."""
    data = json.loads(raw)
    lines = []
    for event in data.get("events") or []:
        segs = event.get("segs") or []
        text = "".join(seg.get("utf8", "") for seg in segs)
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            lines.append(text)
    return " ".join(lines)


def lang_base(code: str) -> str:
    """'en-US-orig' -> 'en', 'es' -> 'es'."""
    return code.removesuffix("-orig").split("-")[0].lower()


def original_language(info: dict) -> str | None:
    """
    Idioma del audio original del vídeo. Hace falta porque YouTube dobla automáticamente muchos
    vídeos con voces de IA, y cada pista doblada trae sus propios subtítulos "xx-orig".
    """
    for fmt in info.get("formats") or []:
        note = fmt.get("format_note") or ""
        if fmt.get("language") and ((fmt.get("language_preference") or 0) >= 10 or "original" in note):
            return fmt["language"]
    return info.get("language")


def fetch_transcript(url: str, langs: list[str]) -> tuple[dict, str | None, str]:
    """
    Extrae info del vídeo y su transcripción.
    Devuelve (info, texto|None, descripción de la fuente).
    """
    with yt_dlp.YoutubeDL(ydl_base_opts()) as ydl:
        info = with_retries("Info del vídeo", lambda: ydl.extract_info(url, download=False))
        # "live_chat" aparece como subtítulo manual en los directos, pero es la repetición del chat
        manual = {k: v for k, v in (info.get("subtitles") or {}).items() if k != "live_chat"}
        auto = info.get("automatic_captions") or {}
        orig = original_language(info)
        orig_base = lang_base(orig) if orig else None
        orig_tracks = [k for k in auto if k.endswith("-orig")]
        # Los "xx-orig" de otros idiomas son el reconocimiento de voz de los doblajes de IA
        # (transcripción de una traducción automática): nunca los usamos.
        if orig_base:
            asr_tracks = [k for k in orig_tracks if lang_base(k) == orig_base]
        else:
            asr_tracks = orig_tracks if len(orig_tracks) == 1 else []

        def matching(table: dict, lang: str) -> list[str]:
            return [k for k in table if not k.endswith("-orig") and lang_base(k) == lang_base(lang)]

        # Preferencia: manuales en el idioma original > manuales en nuestros idiomas >
        # automáticos del audio original > traducciones automáticas de ese audio > cualquier manual
        candidates: list[tuple[str, str]] = []
        for lang in ([orig_base] if orig_base else []) + langs:
            candidates += [("manual", k) for k in matching(manual, lang)]
        candidates += [("auto", k) for k in asr_tracks]
        for lang in ([orig_base] if orig_base else []) + langs:
            candidates += [("auto", k) for k in matching(auto, lang)]
        candidates += [("manual", k) for k in manual]

        seen: set[tuple[str, str]] = set()
        for kind, key in candidates:
            if (kind, key) in seen:
                continue
            seen.add((kind, key))
            formats = (manual if kind == "manual" else auto)[key]
            fmt = next((f for f in formats if f.get("ext") == "json3" and f.get("url")), None)
            if not fmt:
                continue
            try:
                raw = with_retries(f"Subtítulo {kind}/{key}", lambda: ydl.urlopen(fmt["url"]).read(), attempts=2)
                text = parse_json3(raw)
            except Exception as exc:  # noqa: BLE001
                log.warning("No se pudo descargar subtítulo %s/%s: %s", kind, key, short_error(exc))
                continue
            if text:
                return info, text, f"{kind}/{key}"
    return info, None, ""


# --------------------------------------------------------------------------- resumen


def clean_summary(text: str) -> str:
    # Los modelos Qwen "thinking" pueden colar el razonamiento entre <think>...</think>
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    # Quitar restos de Markdown que el modelo insiste en usar
    text = re.sub(r"^\s*[\*•]\s+", "- ", text, flags=re.M)
    text = re.sub(r"^#+\s*", "", text, flags=re.M)
    text = text.replace("**", "")
    return text.strip()


def summarize(transcript: str, title: str, channel: str, duration_min: float | None, instructions: str) -> str:
    if len(transcript) > MAX_TRANSCRIPT_CHARS:
        log.warning("Transcripción muy larga (%d chars), se recorta", len(transcript))
        transcript = transcript[:MAX_TRANSCRIPT_CHARS]

    client = OpenAI(base_url=CONFIG["hetzner_base_url"], api_key=CONFIG["hetzner_api_key"],
                    max_retries=NETWORK_RETRIES, timeout=180)

    system_prompt = (
        f"Eres un asistente que resume vídeos de YouTube a partir de su transcripción. "
        f"Responde siempre en {CONFIG['summary_language']}. "
        "El resultado se enviará por Telegram como texto plano: NO uses Markdown "
        "(nada de **, #, ``` ni tablas). Usa el guion '-' para las listas. "
        "Sé fiel al contenido, no inventes nada y no añadas opiniones propias.\n\n"
        f"Instrucciones sobre el enfoque y la estructura del resumen:\n{instructions}"
    )
    meta = f"Canal: {channel}\nTítulo: {title}"
    if duration_min:
        meta += f"\nDuración: {duration_min:.0f} min"
    user_prompt = f"{meta}\n\nTranscripción:\n\"\"\"\n{transcript}\n\"\"\""

    response = client.chat.completions.create(
        model=CONFIG["hetzner_model"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=2000,
        # Qwen3 es un modelo "thinking": sin esto gasta todos los tokens razonando y devuelve content vacío
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    choice = response.choices[0]
    text = clean_summary(choice.message.content or "")
    if not text:
        raise RuntimeError(f"El modelo devolvió una respuesta vacía (finish_reason={choice.finish_reason})")
    return text


# --------------------------------------------------------------------------- almacenamiento
#
# state.json          estado interno: vídeos vistos, pendientes, a quién se ha enviado cada vídeo,
#                     conversaciones abiertas (preguntas con botones) y offset de Telegram.
# subscriptions.json  suscripciones de cada chat (usuario, grupo o canal) con su prompt.
#
# Se accede desde varios hilos (resúmenes en paralelo), así que todo pasa por STORE_LOCK.


def load_json(path: Path, default: dict) -> dict:
    if path.exists():
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
    else:
        data = {}
    for key, value in default.items():
        data.setdefault(key, json.loads(json.dumps(value)))
    return data


def save_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    tmp.replace(path)


STORE_LOCK = threading.RLock()
STATE_DEFAULT = {
    "processed": {},       # video_id -> info (ya entregado a los suscriptores o descartado)
    "pending": {},         # video_id -> primera vez visto sin subtítulos (ISO)
    "sent": {},            # video_id -> [chat_ids] que ya lo han recibido (evita duplicados)
    "conversations": {},   # conv_id -> pregunta pendiente de respuesta
    "telegram_offset": 0,
}
SUBS_DEFAULT = {
    "default_prompt": None,  # si es None se usa DEFAULT_INSTRUCTIONS
    "chats": {},             # chat_id -> {title, subscriptions: {channel_id: {id, name, handle, url, tabs, prompt}}}
}


@contextmanager
def state_tx():
    with STORE_LOCK:
        data = load_json(CONFIG["state_file"], STATE_DEFAULT)
        yield data
        save_json(CONFIG["state_file"], data)


@contextmanager
def subs_tx():
    with STORE_LOCK:
        data = load_json(CONFIG["subscriptions_file"], SUBS_DEFAULT)
        yield data
        save_json(CONFIG["subscriptions_file"], data)


def read_state() -> dict:
    with STORE_LOCK:
        return load_json(CONFIG["state_file"], STATE_DEFAULT)


def read_subs() -> dict:
    with STORE_LOCK:
        return load_json(CONFIG["subscriptions_file"], SUBS_DEFAULT)


def chat_subscriptions(chat_id: int | str, subs: dict | None = None) -> dict:
    subs = subs or read_subs()
    return subs["chats"].get(str(chat_id), {}).get("subscriptions", {})


def default_prompt() -> str:
    return read_subs().get("default_prompt") or DEFAULT_INSTRUCTIONS


def subscribers_of(channel_id: str) -> list[tuple[str, dict]]:
    return [(chat_id, entry["subscriptions"][channel_id])
            for chat_id, entry in read_subs()["chats"].items()
            if channel_id in entry.get("subscriptions", {})]


def watched_channels() -> dict:
    """Canales con al menos un suscriptor, con la unión de las pestañas que piden."""
    channels: dict[str, dict] = {}
    for entry in read_subs()["chats"].values():
        for channel_id, sub in entry.get("subscriptions", {}).items():
            merged = channels.setdefault(channel_id, {k: v for k, v in sub.items() if k != "prompt"} | {"tabs": []})
            merged["tabs"] += [t for t in channel_tabs(sub) if t not in merged["tabs"]]
    return channels


def record_sent(video_id: str, chat_id: int | str) -> None:
    with state_tx() as st:
        sent = st["sent"].setdefault(video_id, [])
        if str(chat_id) not in sent:
            sent.append(str(chat_id))


def drop_chat(chat_id: int | str, reason: str) -> None:
    """El bot ya no puede escribir en ese chat (expulsado/bloqueado): borramos sus suscripciones."""
    with subs_tx() as subs:
        entry = subs["chats"].pop(str(chat_id), None)
    if entry and entry.get("subscriptions"):
        names = ", ".join(s["name"] for s in entry["subscriptions"].values())
        report(f"🧹 Borradas las suscripciones de «{entry.get('title') or chat_id}» ({chat_id}): {reason}\n{names}")


def migrate_chat(old_id: int, new_id: int) -> None:
    """Un grupo que pasa a supergrupo cambia de chat_id."""
    with subs_tx() as subs:
        if str(old_id) in subs["chats"]:
            subs["chats"][str(new_id)] = subs["chats"].pop(str(old_id))


def migrate_legacy_channels() -> None:
    """channels.json (versión de un solo usuario) -> subscriptions.json, asignado a TELEGRAM_CHAT_ID."""
    legacy = CONFIG["legacy_channels_file"]
    if CONFIG["subscriptions_file"].exists() or not legacy.exists() or not CONFIG["admin_chat_id"]:
        return
    with legacy.open(encoding="utf-8") as fh:
        old = json.load(fh)
    with subs_tx() as subs:
        subs["default_prompt"] = old.get("default_prompt")
        subs["chats"][str(CONFIG["admin_chat_id"])] = {
            "title": "migrado de channels.json",
            "subscriptions": old.get("channels", {}),
        }
    log.info("Migradas %d suscripciones de %s al chat %s", len(old.get("channels", {})), legacy, CONFIG["admin_chat_id"])


# --------------------------------------------------------------------------- telegram: API


class TelegramError(RuntimeError):
    """Respuesta de la API de Telegram con ok=false (no es un error de red)."""


_THREAD_LOCAL = threading.local()
BOT_USERNAME = ""


def _telegram_session() -> requests.Session:
    session = getattr(_THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        # Reintentos a nivel de conexión (DNS, TLS, reset, timeout) y de códigos 5xx/429, con backoff
        retry = Retry(total=NETWORK_RETRIES, connect=NETWORK_RETRIES, read=NETWORK_RETRIES,
                      backoff_factor=NETWORK_BACKOFF_SECONDS, status_forcelist=(429, 500, 502, 503, 504),
                      allowed_methods=frozenset({"POST"}), raise_on_status=False)
        session.mount("https://", HTTPAdapter(max_retries=retry))
        _THREAD_LOCAL.session = session  # requests.Session no es thread-safe: una por hilo
    return session


def telegram_api(method: str, **payload) -> dict:
    url = f"https://api.telegram.org/bot{CONFIG['telegram_token']}/{method}"
    # (timeout de conexión, timeout de lectura): la lectura debe superar el long polling de getUpdates
    timeout = (15, TELEGRAM_LONG_POLL_SECONDS + 15)
    resp = _telegram_session().post(url, json=payload, timeout=timeout)
    data = resp.json()
    if not data.get("ok"):
        raise TelegramError(f"Telegram {method} falló: {data.get('description', data)}")
    return data["result"]


def is_gone_error(exc: BaseException) -> bool:
    """El bot fue expulsado/bloqueado o el chat ya no existe."""
    text = str(exc).lower()
    return any(s in text for s in ("forbidden", "chat not found", "bot was kicked", "bot was blocked"))


def split_message(text: str, limit: int = TELEGRAM_MAX_LEN) -> list[str]:
    chunks = []
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = limit
        chunks.append(text[:cut].rstrip())
        text = text[cut:].lstrip()
    if text:
        chunks.append(text)
    return chunks


def keyboard(rows: list[list[tuple[str, str]]]) -> dict:
    return {"inline_keyboard": [[{"text": text, "callback_data": data} for text, data in row] for row in rows]}


def _preview(enabled: bool) -> dict:
    return {"is_disabled": not enabled, "prefer_small_media": True}


def send_text(chat_id: int | str, text: str, markup: dict | None = None, preview: bool = False) -> int:
    """Envía un mensaje (troceado si hace falta). Devuelve el message_id del primer trozo."""
    first_id = None
    chunks = split_message(text)
    for i, chunk in enumerate(chunks):
        payload = {"chat_id": chat_id, "text": chunk, "link_preview_options": _preview(preview and i == 0)}
        if markup and i == len(chunks) - 1:
            payload["reply_markup"] = markup
        result = telegram_api("sendMessage", **payload)
        first_id = first_id or result["message_id"]
    return first_id


def edit_text(chat_id: int | str, message_id: int, text: str, markup: dict | None = None, preview: bool = False) -> bool:
    """Edita un mensaje (sin markup se quitan los botones). Devuelve False si no se pudo."""
    payload = {"chat_id": chat_id, "message_id": message_id, "text": text[:TELEGRAM_MAX_LEN],
               "link_preview_options": _preview(preview)}
    if markup:
        payload["reply_markup"] = markup
    try:
        telegram_api("editMessageText", **payload)
        return True
    except TelegramError as exc:
        if "message is not modified" in str(exc):
            return True
        log.warning("No se pudo editar el mensaje %s en %s: %s", message_id, chat_id, short_error(exc))
        return False


def delete_message(chat_id: int | str, message_id: int) -> None:
    try:
        telegram_api("deleteMessage", chat_id=chat_id, message_id=message_id)
    except Exception as exc:  # noqa: BLE001
        log.warning("No se pudo borrar el mensaje %s en %s: %s", message_id, chat_id, short_error(exc))


def answer_callback(callback_id: str, text: str | None = None) -> None:
    try:
        telegram_api("answerCallbackQuery", callback_query_id=callback_id, **({"text": text} if text else {}))
    except Exception as exc:  # noqa: BLE001
        log.debug("answerCallbackQuery falló: %s", short_error(exc))


def is_chat_admin(chat_id: int, user_id: int | None) -> bool:
    if not user_id:
        return False
    try:
        member = telegram_api("getChatMember", chat_id=chat_id, user_id=user_id)
    except Exception:  # noqa: BLE001
        return False
    return member.get("status") in ("creator", "administrator")


BOT_COMMANDS = [
    ("video", "Resume un vídeo de YouTube"),
    ("suscribirse", "Recibe un resumen de cada vídeo nuevo de un canal"),
    ("suscripciones", "Ver, editar o borrar tus suscripciones"),
    ("ayuda", "Lista de comandos"),
]


def setup_bot() -> None:
    global BOT_USERNAME
    BOT_USERNAME = telegram_api("getMe").get("username") or ""
    try:
        telegram_api("setMyCommands", commands=[{"command": c, "description": d} for c, d in BOT_COMMANDS])
    except Exception as exc:  # noqa: BLE001
        log.warning("No se pudo registrar el menú de comandos: %s", short_error(exc))


class ProgressMessage:
    """
    Un mensaje de Telegram que se va editando con el estado del proceso
    y que finalmente se convierte en el resumen.
    """

    def __init__(self, chat_id: int | str, url: str, title: str | None = None):
        self.chat_id = chat_id
        self.url = url
        self.title = title
        self.message_id: int | None = None

    def _header(self) -> str:
        return f"🎬 {self.title}\n{self.url}" if self.title else f"🎬 {self.url}"

    def _render(self, text: str, preview: bool) -> None:
        if self.message_id is not None and edit_text(self.chat_id, self.message_id, text, preview=preview):
            return
        # Primer mensaje, o el anterior ya no se puede editar (p. ej. lo han borrado)
        self.message_id = send_text(self.chat_id, text, preview=preview)

    def update(self, status: str) -> None:
        # Sin previsualización mientras trabaja, para que el mensaje se vea compacto
        self._render(f"{self._header()}\n\n⏳ {status}", preview=False)

    def finish(self, summary: str) -> None:
        chunks = split_message(f"{self._header()}\n\n{summary}")
        self._render(chunks[0], preview=True)
        for chunk in chunks[1:]:
            send_text(self.chat_id, chunk)

    def fail(self, reason: str) -> None:
        self._render(f"{self._header()}\n\n❌ {reason}", preview=False)


def print_chat_ids() -> None:
    updates = telegram_api("getUpdates")
    seen = {}
    for upd in updates:
        msg = upd.get("message") or upd.get("channel_post") or {}
        chat = msg.get("chat")
        if chat:
            seen[chat["id"]] = chat.get("title") or chat.get("username") or chat.get("first_name")
    if not seen:
        print("No hay mensajes recientes. Escribe algo al bot (o publica en el canal donde es admin) y vuelve a ejecutar.")
        return
    print("Chats encontrados (usa el id en TELEGRAM_CHAT_ID):")
    for chat_id, name in seen.items():
        print(f"  {chat_id}\t{name}")


# --------------------------------------------------------------------------- notificaciones al admin


def clip(text: str | None, limit: int = 300) -> str:
    if not text:
        return ""
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def chat_label(chat: dict) -> str:
    if chat.get("title"):
        return chat["title"]
    name = " ".join(filter(None, [chat.get("first_name"), chat.get("last_name")]))
    return name or (f"@{chat['username']}" if chat.get("username") else str(chat.get("id")))


def describe_chat(chat: dict) -> str:
    kind = {"private": "privado", "group": "grupo", "supergroup": "grupo", "channel": "canal"}.get(chat.get("type"), "chat")
    if kind == "privado":
        return "privado"
    return f"{kind} «{chat_label(chat)}» ({chat.get('id')})"


def describe_actor(chat: dict, user: dict | None) -> str:
    if not user or chat.get("type") == "channel":
        return describe_chat(chat)
    name = " ".join(filter(None, [user.get("first_name"), user.get("last_name")])) or "?"
    who = f"{name}{' @' + user['username'] if user.get('username') else ''} ({user.get('id')})"
    return who if chat.get("type") == "private" else f"{who} en {describe_chat(chat)}"


def report(text: str, chat_id: int | str | None = None) -> None:
    """Notifica una acción en el chat de administración (salvo que la acción ocurra en ese mismo chat)."""
    admin = CONFIG["admin_chat_id"]
    log.info("REPORT %s", text.replace("\n", " | "))
    if not admin or (chat_id is not None and str(chat_id) == str(admin)):
        return
    try:
        send_text(admin, text)
    except Exception as exc:  # noqa: BLE001
        log.warning("No se pudo notificar al chat de administración: %s", short_error(exc))


# --------------------------------------------------------------------------- trabajos en segundo plano

EXECUTOR: ThreadPoolExecutor | None = None


def submit(name: str, fn, *args) -> Future | None:
    """Ejecuta fn en el pool de hilos (o directamente si no hay pool) registrando los errores."""
    def run():
        try:
            fn(*args)
        except Exception as exc:  # noqa: BLE001
            log_error(f"Error en {name}", exc)
    if EXECUTOR is None:
        run()
        return None
    return EXECUTOR.submit(run)


# --------------------------------------------------------------------------- resúmenes


def resolve_instructions(chat_id: int | str, channel_id: str | None, prompt: str | None) -> tuple[str, str]:
    """(instrucciones, etiqueta). Sin prompt explícito: el de la suscripción del chat a ese canal, o el por defecto."""
    if prompt:
        return prompt, "prompt personalizado"
    sub = chat_subscriptions(chat_id).get(channel_id or "")
    if sub and sub.get("prompt"):
        return sub["prompt"], f"prompt de tu suscripción a {sub['name']}"
    return default_prompt(), "prompt por defecto"


def transcript_problem(info: dict, transcript: str | None) -> str | None:
    if info.get("live_status") in ("is_live", "is_upcoming"):
        return "Es un directo en curso o programado; todavía no tiene transcripción."
    if not transcript:
        return "El vídeo no tiene subtítulos disponibles (aún)."
    return None


def deliver(chat_id: int | str, video_id: str, info: dict, transcript: str, prompt: str | None,
            progress: ProgressMessage | None = None, cache: dict | None = None) -> None:
    """Resume (o reutiliza un resumen de `cache` con las mismas instrucciones) y lo envía al chat."""
    progress = progress or ProgressMessage(chat_id, video_url(video_id))
    progress.title = info.get("title") or progress.title
    instructions, label = resolve_instructions(chat_id, info.get("channel_id"), prompt)
    summary = cache.get(instructions) if cache is not None else None
    if summary is None:
        duration_min = (info.get("duration") or 0) / 60 or None
        log.info("Resumiendo %s para %s (%d chars, %s)…", video_id, chat_id, len(transcript), label)
        if progress.message_id is not None:
            details = f"{duration_min:.0f} min, " if duration_min else ""
            progress.update(f"Resumiendo ({details}{len(transcript) // 1000}k caracteres, {label})…")
        try:
            summary = summarize(transcript, progress.title or video_id, info.get("channel") or "", duration_min, instructions)
        except Exception as exc:
            if progress.message_id is not None:
                progress.fail(f"Error al resumir: {short_error(exc)}")
            raise
        if cache is not None:
            cache[instructions] = summary
    progress.finish(summary)
    record_sent(video_id, chat_id)


def summarize_for_chat(chat_id: int | str, video_id: str, prompt: str | None, actor: str) -> None:
    """Petición manual (/video o ejemplos): mensaje de progreso desde el principio."""
    url = video_url(video_id)
    progress = ProgressMessage(chat_id, url)
    progress.update("Obteniendo subtítulos…")
    try:
        info, transcript, source = fetch_transcript(url, CONFIG["subtitle_langs"])
    except Exception as exc:  # noqa: BLE001
        progress.fail(f"No se pudo acceder al vídeo: {short_error(exc)}")
        report(f"❌ No se pudo acceder a {url} ({actor}): {short_error(exc)}", chat_id)
        return
    problem = transcript_problem(info, transcript)
    if problem:
        progress.fail(problem)
        report(f"⚠️ {url} ({actor}): {problem}", chat_id)
        return
    log.info("Transcripción de %s: %s", video_id, source)
    try:
        deliver(chat_id, video_id, info, transcript, prompt, progress)
    except Exception as exc:  # noqa: BLE001
        log_error(f"Error resumiendo {video_id}", exc)
        report(f"❌ Error resumiendo {url} ({actor}): {short_error(exc)}", chat_id)


def run_examples(chat_id: int | str, video_ids: list[str], actor: str) -> None:
    for video_id in reversed(video_ids):  # del más antiguo al más reciente
        summarize_for_chat(chat_id, video_id, None, actor)


# --------------------------------------------------------------------------- vigilancia de canales


def deliver_to_subscribers(channel: dict, video: dict, info: dict, transcript: str) -> tuple[int, bool]:
    """Envía el resumen a cada suscriptor con su prompt. Devuelve (enviados, hubo_fallos_reintentables)."""
    already = set(read_state()["sent"].get(video["id"], []))
    cache: dict[str, str] = {}  # mismo prompt -> mismo resumen, solo una llamada al LLM
    sent, retry = 0, False
    for chat_id, _sub in subscribers_of(channel["id"]):
        if chat_id in already:
            continue
        try:
            deliver(chat_id, video["id"], info, transcript, None, cache=cache)
            sent += 1
        except TelegramError as exc:
            if is_gone_error(exc):
                drop_chat(chat_id, short_error(exc))
            else:
                log_error(f"No se pudo enviar {video['id']} a {chat_id}", exc)
        except Exception as exc:  # noqa: BLE001
            # Fallo del LLM o de red: se reintenta en la siguiente pasada (sin duplicar a quien ya lo recibió)
            log_error(f"Error resumiendo {video['id']} para {chat_id}", exc)
            retry = True
    return sent, retry


def check_one_channel(channel: dict, now: datetime) -> int:
    """Una pasada por un canal. Devuelve el nº de vídeos nuevos entregados."""
    videos = latest_channel_videos(channel, CONFIG["check_latest_n"])
    if not videos:
        log.warning("No se encontraron vídeos en %s (%s)", channel["url"], "/".join(channel_tabs(channel)))
        return 0
    state = read_state()
    new_videos = [v for v in videos if v["id"] not in state["processed"]]
    new_videos.reverse()  # los más antiguos primero, para que lleguen en orden cronológico
    log.info("[%s] %d vídeos recientes, %d nuevos", channel["name"], len(videos), len(new_videos))

    done = 0
    for video in new_videos:
        if done >= CONFIG["max_per_run"]:
            log.info("[%s] Alcanzado MAX_VIDEOS_PER_RUN, el resto quedará para la siguiente pasada", channel["name"])
            break

        first_seen = state["pending"].get(video["id"])
        if first_seen and now - datetime.fromisoformat(first_seen) > timedelta(hours=CONFIG["give_up_hours"]):
            log.warning("Descartando %s: sin subtítulos tras %.0f h", video["id"], CONFIG["give_up_hours"])
            with state_tx() as st:
                st["processed"][video["id"]] = {"title": video["title"], "gave_up": True, "at": now.isoformat()}
                st["pending"].pop(video["id"], None)
            continue

        try:
            info, transcript, source = fetch_transcript(video["url"], CONFIG["subtitle_langs"])
        except Exception as exc:  # noqa: BLE001
            log_error(f"Error obteniendo {video['id']}; se reintentará en la siguiente pasada", exc)
            continue
        problem = transcript_problem(info, transcript)
        if problem:
            log.info("[%s] %s: %s Se reintentará más tarde.", channel["name"], video["id"], problem)
            with state_tx() as st:
                st["pending"].setdefault(video["id"], now.isoformat())
            continue

        log.info("[%s] Transcripción de %s: %s", channel["name"], video["id"], source)
        sent, retry = deliver_to_subscribers(channel, video, info, transcript)
        if not retry:
            with state_tx() as st:
                st["processed"][video["id"]] = {"title": info.get("title") or video["title"],
                                                "channel": channel["name"], "at": now.isoformat()}
                st["pending"].pop(video["id"], None)
        report(f"📺 Nuevo vídeo de {channel['name']}: {info.get('title') or video['title']}\n{video['url']}\n"
               f"Enviado a {sent} chat(s){' (con errores, se reintentará)' if retry else ''}")
        done += 1
    return done


def check_channels() -> None:
    """Una pasada por todos los canales con suscriptores."""
    channels = watched_channels()
    if not channels:
        log.info("No hay suscripciones todavía")
        return
    now = datetime.now(timezone.utc)
    total = 0
    for channel in channels.values():
        try:
            total += check_one_channel(channel, now)
        except Exception as exc:  # noqa: BLE001
            log_error(f"Error comprobando el canal {channel.get('name')}", exc)
    log.info("Pasada terminada: %d canal(es), %d vídeo(s) nuevo(s)", len(channels), total)


# --------------------------------------------------------------------------- conversaciones
#
# Una conversación es una pregunta del bot con botones (y, en algunos casos, a la espera de un texto):
#   video        ¿con qué prompt resumo este vídeo?
#   subscribe    ¿con qué prompt resumo los vídeos de este canal?
#   edit_prompt  nuevo prompt para una suscripción
#   examples     ¿resumo los últimos vídeos como ejemplo?  (sin respuesta en 30 min -> se borra)

AWAITING_TEXT = ("video", "subscribe", "edit_prompt")


def prompt_buttons(conv_id: str) -> list[list[tuple[str, str]]]:
    return [[("✅ Usar prompt por defecto", f"c:{conv_id}:default")], [("✖️ Cancelar", f"c:{conv_id}:cancel")]]


def prompt_question(chat: dict, header: str, extra: str = "") -> str:
    how = "Escríbeme" if chat.get("type") in ("private", "channel") else "Respóndeme a este mensaje con"
    return (f"{header}\n\n{extra}¿Cómo quieres el resumen? {how} las instrucciones "
            "(p. ej. «enfocado a un trader: primero los pronósticos, luego la justificación») "
            "o usa el prompt por defecto.")


def start_conversation(kind: str, chat: dict, user_id: int | None, data: dict, text: str,
                       buttons, ttl: timedelta, message_id: int | None = None) -> None:
    chat_id = chat["id"]
    conv_id = secrets.token_hex(4)
    replaced = []
    if kind in AWAITING_TEXT:
        # Solo una pregunta abierta por persona y chat: si lanza otro comando, la anterior se cancela
        with state_tx() as st:
            for cid, conv in list(st["conversations"].items()):
                if conv["kind"] in AWAITING_TEXT and conv["chat"]["id"] == chat_id and conv["user_id"] == user_id:
                    replaced.append(st["conversations"].pop(cid))
    for conv in replaced:
        edit_text(chat_id, conv["message_id"], "✖️ Cancelado: has lanzado otro comando.")

    markup = keyboard(buttons(conv_id))
    if message_id is None or not edit_text(chat_id, message_id, text, markup):
        message_id = send_text(chat_id, text, markup)
    now = datetime.now(timezone.utc)
    with state_tx() as st:
        st["conversations"][conv_id] = {
            "id": conv_id, "kind": kind, "user_id": user_id, "data": data, "message_id": message_id,
            "chat": {"id": chat_id, "type": chat.get("type"), "title": chat_label(chat)},
            "created_at": now.isoformat(), "expires_at": (now + ttl).isoformat(),
        }


def pop_conversation(conv_id: str) -> dict | None:
    with state_tx() as st:
        return st["conversations"].pop(conv_id, None)


def find_text_conversation(chat_id: int, user_id: int | None, reply_to: int | None, allow_unreplied: bool) -> dict | None:
    convs = [c for c in read_state()["conversations"].values()
             if c["kind"] in AWAITING_TEXT and c["chat"]["id"] == chat_id]
    if reply_to:
        for conv in convs:
            if conv["message_id"] == reply_to:
                return conv
    if not allow_unreplied:
        return None
    mine = sorted((c for c in convs if c["user_id"] == user_id), key=lambda c: c["created_at"])
    return mine[-1] if mine else None


def can_answer(conv_or_chat: dict, chat: dict, user: dict | None) -> bool:
    """En grupos solo responde quien lanzó el comando; en canales, solo sus administradores."""
    user_id = (user or {}).get("id")
    if conv_or_chat.get("user_id") is not None:
        return user_id == conv_or_chat["user_id"]
    if chat.get("type") == "channel":
        return is_chat_admin(chat["id"], user_id)
    return True


def expire_conversations() -> None:
    now = datetime.now(timezone.utc)
    with state_tx() as st:
        expired = [st["conversations"].pop(cid) for cid, conv in list(st["conversations"].items())
                   if datetime.fromisoformat(conv["expires_at"]) < now]
    for conv in expired:
        if conv["kind"] == "examples":
            delete_message(conv["chat"]["id"], conv["message_id"])  # sin respuesta = Omitir
            log.info("Ejemplos de %s omitidos en %s por falta de respuesta", conv["data"].get("channel_name"), conv["chat"]["id"])
        else:
            edit_text(conv["chat"]["id"], conv["message_id"], "⌛ Esta pregunta ha caducado. Vuelve a lanzar el comando.")


def answer_prompt(conv: dict, prompt: str | None, actor: str) -> None:
    """El usuario ha elegido prompt (texto) o ha pulsado «Usar prompt por defecto» (None)."""
    chat, data = conv["chat"], conv["data"]
    chat_id = chat["id"]
    choice = f"📝 Prompt: {clip(prompt, 1000)}" if prompt else "📝 Prompt por defecto"
    edit_text(chat_id, conv["message_id"], f"{data.get('header', '')}\n\n{choice}".strip())

    if conv["kind"] == "video":
        report(f"🎬 Resumen pedido por {actor}\n{video_url(data['video_id'])}\n📝 {clip(prompt) or 'prompt por defecto'}", chat_id)
        submit("resumen", summarize_for_chat, chat_id, data["video_id"], prompt, actor)
    elif conv["kind"] == "subscribe":
        submit("suscripción", complete_subscription, chat, conv["user_id"], data["channel"], prompt, actor)
    elif conv["kind"] == "edit_prompt":
        with subs_tx() as subs:
            sub = chat_subscriptions(chat_id, subs).get(data["channel_id"])
            if sub:
                sub["prompt"] = prompt
        if not sub:
            send_text(chat_id, f"Ya no estás suscrito a {data['name']}.")
            return
        send_text(chat_id, f"✅ Prompt de {data['name']} actualizado" + ("." if prompt else ": se usará el prompt por defecto."))
        report(f"✏️ {actor} cambió el prompt de {data['name']}\n📝 {clip(prompt) or 'por defecto'}", chat_id)


# --------------------------------------------------------------------------- comandos

HELP_COMMANDS = ("/ayuda", "/start", "/help")
HELP_TEXT = """Te resumo vídeos de YouTube y te aviso cuando los canales que sigues suben algo nuevo.

/video <url> — resume un vídeo. Te preguntaré cómo quieres el resumen (o puedes usar el prompt por defecto). Si escribes las instrucciones después de la URL, me salto la pregunta.

/suscribirse <url del canal> — te envío un resumen de cada vídeo nuevo del canal. Para seguir sus directos, usa la URL acabada en /streams.

/suscripciones — ver tus suscripciones, cambiar su prompt o borrarlas.

/ayuda — esta ayuda."""


def parse_command(text: str) -> tuple[str, str] | None:
    """'/video@MiBot url' -> ('/video', 'url'). None si el comando va dirigido a otro bot."""
    parts = text.strip().split(maxsplit=1)
    command, _, target = parts[0].partition("@")
    if target and BOT_USERNAME and target.lower() != BOT_USERNAME.lower():
        return None
    return command.lower(), (parts[1].strip() if len(parts) > 1 else "")


def instructions_from_text(text: str) -> str | None:
    """El texto que acompaña a la URL son instrucciones para el resumen."""
    # Quitamos la URL completa, incluidos parámetros como &t=120s o &list=... que van tras el ID
    extra = re.sub(YOUTUBE_URL_RE.pattern + r"\S*", "", text)
    return re.sub(r"[ \t]+", " ", extra).strip() or None


def cmd_video(chat: dict, user_id: int | None, arg: str, actor: str) -> None:
    ids = extract_video_ids(arg)
    if not ids:
        send_text(chat["id"], "Uso: /video <url de YouTube> [instrucciones opcionales]")
        return
    video_id = ids[0]
    url = video_url(video_id)
    prompt = instructions_from_text(arg)
    if prompt:
        report(f"🎬 Resumen pedido por {actor}\n{url}\n📝 {clip(prompt)}", chat["id"])
        submit("resumen", summarize_for_chat, chat["id"], video_id, prompt, actor)
        return
    report(f"🎬 /video — {actor}\n{url}", chat["id"])
    header = f"🎬 {url}"
    start_conversation("video", chat, user_id, {"video_id": video_id, "header": header},
                       prompt_question(chat, header), prompt_buttons, PROMPT_TIMEOUT)


def cmd_subscribe(chat: dict, user_id: int | None, arg: str, actor: str) -> None:
    if not arg:
        send_text(chat["id"], "Uso: /suscribirse <url del canal de YouTube> [instrucciones opcionales]\n"
                              "Ejemplo: /suscribirse https://www.youtube.com/@MeetKevin")
        return
    parts = arg.split(maxsplit=1)
    ref, prompt = parts[0], (parts[1].strip() if len(parts) > 1 else None)
    report(f"➕ /suscribirse — {actor}\n{ref}", chat["id"])
    message_id = send_text(chat["id"], "🔎 Buscando el canal…")
    submit("suscripción", prepare_subscription, chat, user_id, ref, prompt, message_id, actor)


def prepare_subscription(chat: dict, user_id: int | None, ref: str, prompt: str | None, message_id: int, actor: str) -> None:
    chat_id = chat["id"]
    try:
        channel = resolve_channel(ref)
    except Exception as exc:  # noqa: BLE001
        edit_text(chat_id, message_id, f"❌ No he encontrado ese canal ({short_error(exc)}).\n\n"
                                       "Uso: /suscribirse <url del canal>, p. ej. https://www.youtube.com/@MeetKevin")
        return
    if channel["id"] in chat_subscriptions(chat_id):
        edit_text(chat_id, message_id, f"Ya estás suscrito a {channel['name']}. Usa /suscripciones para cambiar el prompt o borrarla.")
        return
    what = "directos" if channel_tabs(channel) == ["streams"] else "vídeos"
    header = f"📺 {channel['name']} ({what})\n{channel['url']}"
    if prompt:
        edit_text(chat_id, message_id, f"{header}\n\n📝 Prompt: {clip(prompt, 1000)}")
        complete_subscription({"id": chat_id, "type": chat.get("type"), "title": chat_label(chat)},
                              user_id, channel, prompt, actor)
        return
    start_conversation("subscribe", chat, user_id, {"channel": channel, "header": header},
                       prompt_question(chat, header), prompt_buttons, PROMPT_TIMEOUT, message_id=message_id)


def complete_subscription(chat: dict, user_id: int | None, channel: dict, prompt: str | None, actor: str) -> None:
    chat_id = chat["id"]
    already_watched = bool(subscribers_of(channel["id"]))
    with subs_tx() as subs:
        entry = subs["chats"].setdefault(str(chat_id), {"title": chat["title"], "subscriptions": {}})
        entry["title"] = chat["title"]
        entry["subscriptions"][channel["id"]] = channel | {"prompt": prompt}

    latest: list[dict] = []
    try:
        latest = latest_channel_videos(channel, CONFIG["check_latest_n"])
    except Exception as exc:  # noqa: BLE001
        log_error(f"No se pudo listar {channel['name']}", exc)
    if not already_watched:
        # Canal nuevo para el vigilante: lo ya publicado no cuenta como "nuevo" (para eso están los ejemplos).
        # Si otros chats ya lo seguían no tocamos nada, para no quitarles vídeos aún no entregados.
        now = datetime.now(timezone.utc).isoformat()
        with state_tx() as st:
            for v in latest:
                st["processed"].setdefault(v["id"], {"title": v["title"], "seen_on_subscribe": True, "at": now})

    what = "directo" if channel_tabs(channel) == ["streams"] else "vídeo"
    send_text(chat_id, f"✅ Suscrito a {channel['name']}. Te enviaré un resumen de cada {what} nuevo.\n"
                       "Gestiona tus suscripciones con /suscripciones.")
    report(f"✅ Suscripción: {actor} → {channel['name']}\n{channel['url']}\n📝 {clip(prompt) or 'prompt por defecto'}", chat_id)

    examples = latest[:EXAMPLES_COUNT]
    if examples:
        n = len(examples)
        start_conversation(
            "examples", chat, user_id,
            {"channel_name": channel["name"], "video_ids": [v["id"] for v in examples]},
            f"¿Quieres que resuma los últimos {n} vídeos de {channel['name']} como ejemplo?",
            lambda cid: [[(f"▶️ Resumir los últimos {n}", f"c:{cid}:examples"), ("Omitir", f"c:{cid}:skip")]],
            EXAMPLES_TIMEOUT,
        )


def render_subscriptions(chat_id: int | str) -> tuple[str, dict | None]:
    subs = chat_subscriptions(chat_id)
    if not subs:
        return "No tienes suscripciones. Usa /suscribirse <url del canal> para añadir una.", None
    lines = ["📋 Suscripciones:"]
    rows = []
    for i, (channel_id, sub) in enumerate(subs.items(), 1):
        what = "+".join("directos" if t == "streams" else "vídeos" for t in channel_tabs(sub))
        lines.append(f"\n{i}. {sub['name']} ({what})\n{sub['url']}\n📝 {clip(sub.get('prompt'), 200) or 'prompt por defecto'}")
        name = clip(sub["name"], 18)
        rows.append([(f"✏️ {i}. {name}", f"s:e:{channel_id}"), (f"🗑 {i}. {name}", f"s:d:{channel_id}")])
    return "\n".join(lines), keyboard(rows)


def cmd_subscriptions(chat: dict, actor: str) -> None:
    text, markup = render_subscriptions(chat["id"])
    send_text(chat["id"], text, markup)
    report(f"📋 /suscripciones — {actor}", chat["id"])


# --------------------------------------------------------------------------- despacho de updates


def handle_message(msg: dict) -> None:
    chat = msg.get("chat") or {}
    chat_id = chat.get("id")
    if msg.get("migrate_to_chat_id"):
        migrate_chat(chat_id, msg["migrate_to_chat_id"])
        return
    text = (msg.get("text") or msg.get("caption") or "").strip()
    if not chat_id or not text:
        return
    user = msg.get("from") if chat.get("type") != "channel" else None
    user_id = (user or {}).get("id")
    actor = describe_actor(chat, user)

    if text.startswith("/"):
        parsed = parse_command(text)
        if not parsed:
            return
        command, arg = parsed
        if command in HELP_COMMANDS:
            send_text(chat_id, HELP_TEXT)
            report(f"❔ {command} — {actor}", chat_id)
        elif command == "/video":
            cmd_video(chat, user_id, arg, actor)
        elif command == "/suscribirse":
            cmd_subscribe(chat, user_id, arg, actor)
        elif command == "/suscripciones":
            cmd_subscriptions(chat, actor)
        elif chat.get("type") == "private":
            send_text(chat_id, f"No conozco el comando {command}.\n\n{HELP_TEXT}")
        return

    # ¿Es la respuesta a una pregunta de prompt? En grupos solo si responde al mensaje del bot
    # (con el modo privacidad de Telegram el bot no ve el resto de mensajes del grupo).
    reply_to = (msg.get("reply_to_message") or {}).get("message_id")
    conv = find_text_conversation(chat_id, user_id, reply_to, allow_unreplied=chat.get("type") in ("private", "channel"))
    if conv:
        conv = pop_conversation(conv["id"])
        if conv:
            answer_prompt(conv, text, actor)
        return

    if chat.get("type") == "private":
        if extract_video_ids(text):
            cmd_video(chat, user_id, text, actor)  # una URL suelta en privado = /video
        else:
            send_text(chat_id, HELP_TEXT)


def handle_callback(cb: dict) -> None:
    data = cb.get("data") or ""
    msg = cb.get("message") or {}
    chat = msg.get("chat") or {}
    user = cb.get("from")
    actor = describe_actor(chat, user) if chat.get("type") != "channel" else f"{describe_chat(chat)} (admin {(user or {}).get('id')})"

    if data.startswith("c:"):
        _, conv_id, action = data.split(":", 2)
        conv = read_state()["conversations"].get(conv_id)
        if not conv:
            answer_callback(cb["id"], "Esta pregunta ya no está activa.")
            return
        if not can_answer(conv, chat, user):
            answer_callback(cb["id"], "Solo quien lanzó el comando puede responder.")
            return
        conv = pop_conversation(conv_id)
        answer_callback(cb["id"])
        if not conv:
            return
        chat_id = conv["chat"]["id"]
        if action == "cancel":
            edit_text(chat_id, conv["message_id"], "✖️ Cancelado.")
        elif action == "default":
            answer_prompt(conv, None, actor)
        elif action == "examples":
            d = conv["data"]
            edit_text(chat_id, conv["message_id"], f"▶️ Resumiendo los últimos {len(d['video_ids'])} vídeos de {d['channel_name']}…")
            report(f"▶️ {actor} pidió los ejemplos de {d['channel_name']}", chat_id)
            submit("ejemplos", run_examples, chat_id, d["video_ids"], actor)
        elif action == "skip":
            delete_message(chat_id, conv["message_id"])
        return

    if data.startswith("s:"):
        _, action, channel_id = data.split(":", 2)
        chat_id = chat.get("id")
        if not can_answer({}, chat, user):
            answer_callback(cb["id"], "Solo los administradores del canal pueden gestionar las suscripciones.")
            return
        sub = chat_subscriptions(chat_id).get(channel_id)
        if not sub:
            answer_callback(cb["id"], "Esa suscripción ya no existe.")
            text, markup = render_subscriptions(chat_id)
            edit_text(chat_id, msg["message_id"], text, markup)
            return
        if action == "d":
            with subs_tx() as subs:
                chat_subscriptions(chat_id, subs).pop(channel_id, None)
            answer_callback(cb["id"], f"Borrada la suscripción a {sub['name']}")
            text, markup = render_subscriptions(chat_id)
            edit_text(chat_id, msg["message_id"], text, markup)
            report(f"🗑 {actor} borró la suscripción a {sub['name']}", chat_id)
        elif action == "e":
            answer_callback(cb["id"])
            header = f"✏️ Prompt de {sub['name']}"
            current = f"Prompt actual:\n{sub.get('prompt') or '(prompt por defecto)'}\n\n"
            user_id = None if chat.get("type") == "channel" else (user or {}).get("id")
            start_conversation("edit_prompt", chat, user_id, {"channel_id": channel_id, "name": sub["name"], "header": header},
                               prompt_question(chat, header, current), prompt_buttons, PROMPT_TIMEOUT)
        return

    answer_callback(cb["id"])


def handle_membership(update: dict) -> None:
    """El bot ha sido añadido o expulsado de un grupo/canal (o bloqueado en privado)."""
    chat = update.get("chat") or {}
    status = (update.get("new_chat_member") or {}).get("status")
    actor = describe_actor({"type": "private"}, update.get("from"))
    if status in ("member", "administrator"):
        report(f"➕ Bot añadido a {describe_chat(chat)} por {actor} ({status})")
        if chat.get("type") in ("group", "supergroup"):
            send_text(chat["id"], HELP_TEXT)
    elif status in ("left", "kicked"):
        report(f"➖ Bot eliminado de {describe_chat(chat) if chat.get('type') != 'private' else actor} ({status})")
        drop_chat(chat.get("id"), f"bot {status}")


def handle_telegram_updates() -> None:
    state = read_state()
    updates = telegram_api(
        "getUpdates",
        offset=state["telegram_offset"],
        timeout=TELEGRAM_LONG_POLL_SECONDS,
        allowed_updates=["message", "channel_post", "callback_query", "my_chat_member"],
    )
    for upd in updates:
        # Confirmamos el update antes de procesarlo para no repetirlo si algo falla a mitad
        with state_tx() as st:
            st["telegram_offset"] = upd["update_id"] + 1
        try:
            if "callback_query" in upd:
                handle_callback(upd["callback_query"])
            elif "my_chat_member" in upd:
                handle_membership(upd["my_chat_member"])
            elif upd.get("message") or upd.get("channel_post"):
                handle_message(upd.get("message") or upd["channel_post"])
        except Exception as exc:  # noqa: BLE001
            log_error(f"Error atendiendo el update {upd.get('update_id')}", exc)


# --------------------------------------------------------------------------- main


def main() -> None:
    global EXECUTOR
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--once", action="store_true", help="Una sola pasada por los canales y termina (para cron)")
    parser.add_argument("--get-chat-id", action="store_true", help="Muestra los chat_id que han escrito al bot")
    parser.add_argument("--video", metavar="URL", help="Resume un vídeo y lo envía al chat de administración")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(threadName)s %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "httpx2", "httpcore", "urllib3"):  # el SDK de OpenAI usa un fork llamado httpx2
        logging.getLogger(noisy).setLevel(logging.WARNING)

    env("TELEGRAM_BOT_API_KEY", required=True)
    if args.get_chat_id:
        print_chat_ids()
        return

    env("HETZNER_INFERENCE_API_KEY", required=True)
    env("TELEGRAM_CHAT_ID", required=True)
    migrate_legacy_channels()

    if args.video:
        for video_id in extract_video_ids(args.video) or [args.video]:
            summarize_for_chat(CONFIG["admin_chat_id"], video_id, instructions_from_text(args.video), "línea de comandos")
        return

    if args.once:
        check_channels()
        return

    setup_bot()
    EXECUTOR = ThreadPoolExecutor(max_workers=CONFIG["workers"], thread_name_prefix="job")
    log.info("Bot @%s escuchando; canales cada %.0f min; notificaciones a %s",
             BOT_USERNAME, CONFIG["poll_minutes"], CONFIG["admin_chat_id"])
    next_channel_check = 0.0
    channel_job: Future | None = None
    telegram_failures = 0
    while True:
        if time.time() >= next_channel_check and (channel_job is None or channel_job.done()):
            next_channel_check = time.time() + CONFIG["poll_minutes"] * 60
            channel_job = submit("comprobación de canales", check_channels)
        try:
            expire_conversations()
            handle_telegram_updates()  # bloquea hasta TELEGRAM_LONG_POLL_SECONDS si no hay mensajes
            if telegram_failures:
                log.info("Conexión con Telegram recuperada")
            telegram_failures = 0
        except Exception as exc:  # noqa: BLE001
            telegram_failures += 1
            wait = min(5 * 2 ** (telegram_failures - 1), 120)
            log.warning("Telegram no responde (%s). Reintento %d en %ds", short_error(exc), telegram_failures, wait)
            log.debug("Traceback:", exc_info=exc)
            time.sleep(wait)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
