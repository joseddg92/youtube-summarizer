#!/usr/bin/env python3
"""
Vigila uno o varios canales de YouTube, descarga los subtítulos de los vídeos nuevos (vía yt-dlp),
los resume con la Inference API de Hetzner (compatible con OpenAI) y envía el resumen junto con
el enlace por Telegram. Además escucha el bot de Telegram: cualquier URL de YouTube que le envíes
se resume al momento, y desde el propio chat puedes gestionar los canales y el estilo de resumen
de cada uno (/help).

Uso:
    python main.py                # bucle: escucha Telegram y comprueba los canales cada POLL_INTERVAL_MINUTES
    python main.py --once         # una sola pasada por los canales (ideal para cron; no escucha Telegram)
    python main.py --get-chat-id  # ayuda para obtener TELEGRAM_CHAT_ID
    python main.py --video URL    # resume un vídeo concreto (ignora el estado)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
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
    "seed_channel": env("YOUTUBE_CHANNEL_ID"),
    "subtitle_langs": [l.strip() for l in env("SUBTITLE_LANGUAGES", "en,es").split(",") if l.strip()],
    "check_latest_n": int(env("CHECK_LATEST_N", "10")),
    "max_per_run": int(env("MAX_VIDEOS_PER_RUN", "3")),
    "first_run_videos": int(env("FIRST_RUN_VIDEOS", "1")),
    "give_up_hours": float(env("GIVE_UP_AFTER_HOURS", "48")),
    "cookies_file": env("YTDLP_COOKIES_FILE"),
    "cookies_from_browser": env("YTDLP_COOKIES_FROM_BROWSER"),
    "telegram_token": env("TELEGRAM_BOT_API_KEY"),
    "telegram_chat_id": env("TELEGRAM_CHAT_ID"),
    "summary_language": env("SUMMARY_LANGUAGE", "español"),
    "poll_minutes": float(env("POLL_INTERVAL_MINUTES", "30")),
    "state_file": Path(env("STATE_FILE", "state.json")),
    "channels_file": Path(env("CHANNELS_FILE", "channels.json")),
}

# Límite de seguridad para la transcripción (~100k tokens; el modelo admite 262k)
MAX_TRANSCRIPT_CHARS = 400_000
TELEGRAM_MAX_LEN = 4000  # el límite real es 4096
TELEGRAM_LONG_POLL_SECONDS = 25
NETWORK_RETRIES = 3          # reintentos por llamada de red (Telegram, YouTube, Hetzner)
NETWORK_BACKOFF_SECONDS = 3  # espera inicial entre reintentos (se duplica en cada uno)

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


# --------------------------------------------------------------------------- ficheros JSON


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


STATE_DEFAULT = {
    "processed": {},            # video_id -> info
    "pending": {},              # video_id -> primera vez visto (ISO)
    "initialized_channels": [],  # channel_ids cuyo histórico ya se marcó como visto
    "telegram_offset": 0,
}
CHANNELS_DEFAULT = {
    "default_prompt": None,  # si es None se usa DEFAULT_INSTRUCTIONS
    "channels": {},          # channel_id -> {name, handle, url, prompt}
}


def load_state() -> dict:
    return load_json(CONFIG["state_file"], STATE_DEFAULT)


def save_state(state: dict) -> None:
    save_json(CONFIG["state_file"], state)


def load_channels() -> dict:
    return load_json(CONFIG["channels_file"], CHANNELS_DEFAULT)


def save_channels(channels: dict) -> None:
    save_json(CONFIG["channels_file"], channels)


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
    """Convierte una URL/@handle/ID en un registro de canal {id, name, handle, url, tabs}."""
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


def fetch_transcript(url: str, langs: list[str]) -> tuple[dict, str | None, str]:
    """
    Extrae info del vídeo y su transcripción.
    Devuelve (info, texto|None, descripción de la fuente).
    Prioridad: subtítulos manuales > automáticos, en el orden de `langs`.
    """
    with yt_dlp.YoutubeDL(ydl_base_opts()) as ydl:
        info = with_retries("Info del vídeo", lambda: ydl.extract_info(url, download=False))
        # "live_chat" aparece como subtítulo manual en los directos, pero es la repetición del chat
        manual = {k: v for k, v in (info.get("subtitles") or {}).items() if k != "live_chat"}
        auto = info.get("automatic_captions") or {}
        # Orden de preferencia: manuales en nuestros idiomas > automáticos en el idioma ORIGINAL
        # del vídeo (clave "xx-orig"; las demás son traducciones automáticas, peores) > automáticos
        # en nuestros idiomas > cualquier manual.
        candidates: list[tuple[str, str, list]] = []
        for lang in langs:
            candidates += [("manual", k, f) for k, f in manual.items() if k == lang or k.startswith(lang + "-")]
        candidates += [("auto", k, f) for k, f in auto.items() if k.endswith("-orig")]
        for lang in langs:
            candidates += [("auto", k, f) for k, f in auto.items() if k == lang or k.startswith(lang + "-")]
        candidates += [("manual", k, f) for k, f in manual.items()]
        seen: set[tuple[str, str]] = set()
        for kind, key, formats in candidates:
            if (kind, key) in seen:
                continue
            seen.add((kind, key))
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


def instructions_for(channel_id: str | None, channels: dict, extra: str | None = None) -> tuple[str, str]:
    """Devuelve (instrucciones, etiqueta descriptiva) para un vídeo según su canal y las instrucciones puntuales."""
    entry = channels["channels"].get(channel_id or "", {})
    if entry.get("prompt"):
        instructions, label = entry["prompt"], f"prompt de {entry.get('name') or channel_id}"
    elif channels.get("default_prompt"):
        instructions, label = channels["default_prompt"], "prompt por defecto (personalizado)"
    else:
        instructions, label = DEFAULT_INSTRUCTIONS, "prompt por defecto"
    if extra:
        instructions = f"{instructions}\n\nInstrucciones adicionales para este vídeo en concreto (tienen prioridad):\n{extra}"
        label += " + instrucciones del mensaje"
    return instructions, label


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


# --------------------------------------------------------------------------- telegram


class TelegramError(RuntimeError):
    """Respuesta de la API de Telegram con ok=false (no es un error de red)."""


def _telegram_session() -> requests.Session:
    session = requests.Session()
    # Reintentos a nivel de conexión (DNS, TLS, reset, timeout) y de códigos 5xx/429, con backoff
    retry = Retry(total=NETWORK_RETRIES, connect=NETWORK_RETRIES, read=NETWORK_RETRIES,
                  backoff_factor=NETWORK_BACKOFF_SECONDS, status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=frozenset({"POST"}), raise_on_status=False)
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


TELEGRAM_SESSION = _telegram_session()


def telegram_api(method: str, **payload) -> dict:
    url = f"https://api.telegram.org/bot{CONFIG['telegram_token']}/{method}"
    # (timeout de conexión, timeout de lectura): la lectura debe superar el long polling de getUpdates
    timeout = (15, TELEGRAM_LONG_POLL_SECONDS + 15)
    resp = TELEGRAM_SESSION.post(url, json=payload, timeout=timeout)
    data = resp.json()
    if not data.get("ok"):
        raise TelegramError(f"Telegram {method} falló: {data.get('description', data)}")
    return data["result"]


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


def send_message(chat_id: str | int, text: str, preview: bool = True) -> int:
    """Envía un mensaje (troceado si hace falta). Devuelve el message_id del primer trozo."""
    first_id = None
    for i, chunk in enumerate(split_message(text)):
        result = telegram_api(
            "sendMessage",
            chat_id=chat_id,
            text=chunk,
            link_preview_options={"is_disabled": not (preview and i == 0), "prefer_small_media": True},
        )
        first_id = first_id or result["message_id"]
    return first_id


class ProgressMessage:
    """
    Un mensaje de Telegram que se va editando con el estado del proceso
    y que finalmente se convierte en el resumen.
    """

    def __init__(self, chat_id: str | int, url: str, title: str | None = None):
        self.chat_id = chat_id
        self.url = url
        self.title = title
        self.message_id: int | None = None

    def _header(self) -> str:
        return f"🎬 {self.title}\n{self.url}" if self.title else f"🎬 {self.url}"

    def _render(self, text: str, preview: bool) -> None:
        options = {"is_disabled": not preview, "prefer_small_media": True}
        if self.message_id is None:
            result = telegram_api("sendMessage", chat_id=self.chat_id, text=text, link_preview_options=options)
            self.message_id = result["message_id"]
            return
        try:
            telegram_api("editMessageText", chat_id=self.chat_id, message_id=self.message_id,
                         text=text, link_preview_options=options)
        except TelegramError as exc:
            if "message is not modified" not in str(exc):
                raise

    def update(self, status: str) -> None:
        # Sin previsualización mientras trabaja, para que el mensaje se vea compacto
        self._render(f"{self._header()}\n\n⏳ {status}", preview=False)

    def finish(self, summary: str) -> None:
        chunks = split_message(f"{self._header()}\n\n{summary}")
        self._render(chunks[0], preview=True)
        for chunk in chunks[1:]:
            send_message(self.chat_id, chunk, preview=False)

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
        print("No hay mensajes recientes. Escribe algo a tu bot en Telegram y vuelve a ejecutar.")
        return
    print("Chats encontrados (usa el id en TELEGRAM_CHAT_ID):")
    for chat_id, name in seen.items():
        print(f"  {chat_id}\t{name}")


# --------------------------------------------------------------------------- flujo principal


def mark_processed(video_id: str, title: str, **extra) -> None:
    state = load_state()
    state["processed"][video_id] = {"title": title, "at": datetime.now(timezone.utc).isoformat(), **extra}
    state["pending"].pop(video_id, None)
    save_state(state)


def process_video(video: dict, chat_id: str | int, eager: bool, extra_instructions: str | None = None) -> bool:
    """
    Descarga subtítulos, resume y envía. Devuelve True si se completó, False si aún no hay subtítulos.

    eager=True: envía el mensaje de progreso desde el principio (petición manual por Telegram).
    eager=False: solo empieza a escribir en Telegram cuando ya hay transcripción, para no
                 llenar el chat con reintentos de vídeos recién subidos sin subtítulos.
    """
    log.info("Procesando %s — %s", video["id"], video["title"])
    progress = ProgressMessage(chat_id, video["url"], video.get("title") if video.get("title") != video["id"] else None)
    if eager:
        progress.update("Obteniendo subtítulos…")

    try:
        info, transcript, source = fetch_transcript(video["url"], CONFIG["subtitle_langs"])
    except Exception as exc:  # noqa: BLE001
        if eager:
            progress.fail(f"No se pudo acceder al vídeo: {short_error(exc)}")
        raise

    progress.title = info.get("title") or progress.title
    live_status = info.get("live_status")
    if live_status in ("is_live", "is_upcoming"):
        log.info("Vídeo en directo/programado (%s), se reintentará más tarde", live_status)
        if eager:
            progress.fail("Es un directo o un vídeo programado; todavía no tiene transcripción.")
        return False
    if not transcript:
        log.info("Todavía no hay subtítulos disponibles, se reintentará más tarde")
        if eager:
            progress.fail("El vídeo no tiene subtítulos disponibles (aún).")
        return False

    channel_name = info.get("channel") or info.get("uploader") or ""
    instructions, label = instructions_for(info.get("channel_id"), load_channels(), extra_instructions)
    duration_min = (info.get("duration") or 0) / 60 or None
    log.info("Transcripción obtenida (%s, %d chars). Resumiendo con %s usando %s…",
             source, len(transcript), CONFIG["hetzner_model"], label)
    details = f"{duration_min:.0f} min, " if duration_min else ""
    progress.update(f"Resumiendo ({details}{len(transcript) // 1000}k caracteres, {label})…")

    try:
        summary = summarize(transcript, progress.title, channel_name, duration_min, instructions)
    except Exception as exc:  # noqa: BLE001
        progress.fail(f"Error al resumir: {short_error(exc)}")
        raise

    progress.finish(summary)
    log.info("Resumen enviado por Telegram")
    if eager:
        # Resumido a mano: que el vigilante de canales no lo vuelva a enviar
        mark_processed(video["id"], progress.title or video["id"], manual=True)
    return True


def check_one_channel(channel: dict, state: dict, now: datetime) -> int:
    """Una pasada por un canal. Devuelve el nº de vídeos resumidos."""
    videos: list[dict] = []
    for tab in channel_tabs(channel):
        _, tab_videos = fetch_channel(normalize_channel_url(channel["url"], tab), CONFIG["check_latest_n"])
        videos += [v for v in tab_videos if v["id"] not in {x["id"] for x in videos}]
    if not videos:
        log.warning("No se encontraron vídeos en %s (%s)", channel["url"], "/".join(channel_tabs(channel)))
        return 0
    log.info("[%s] %d vídeos recientes (%s)", channel["name"], len(videos), "/".join(channel_tabs(channel)))

    if channel["id"] not in state["initialized_channels"]:
        # Canal nuevo: marcamos como vistos todos menos los N más recientes para no reventar el chat
        for v in videos[CONFIG["first_run_videos"]:]:
            state["processed"].setdefault(v["id"], {"title": v["title"], "skipped_on_first_run": True, "at": now.isoformat()})
        state["initialized_channels"].append(channel["id"])
        save_state(state)

    new_videos = [v for v in videos if v["id"] not in state["processed"]]
    # Los más antiguos primero, para que lleguen por Telegram en orden cronológico
    new_videos.reverse()

    done = 0
    for video in new_videos:
        if done >= CONFIG["max_per_run"]:
            log.info("[%s] Alcanzado MAX_VIDEOS_PER_RUN, el resto quedará para la siguiente pasada", channel["name"])
            break

        first_seen = state["pending"].get(video["id"])
        if first_seen:
            age = now - datetime.fromisoformat(first_seen)
            if age > timedelta(hours=CONFIG["give_up_hours"]):
                log.warning("Descartando %s: sin subtítulos tras %.0f h", video["id"], age.total_seconds() / 3600)
                state["processed"][video["id"]] = {"title": video["title"], "gave_up": True, "at": now.isoformat()}
                state["pending"].pop(video["id"], None)
                save_state(state)
                continue

        try:
            ok = process_video(video, CONFIG["telegram_chat_id"], eager=False)
        except Exception as exc:  # noqa: BLE001
            log_error(f"Error procesando {video['id']}; se reintentará en la siguiente pasada", exc)
            continue

        if ok:
            state["processed"][video["id"]] = {"title": video["title"], "channel": channel["name"], "at": now.isoformat()}
            state["pending"].pop(video["id"], None)
            done += 1
        else:
            state["pending"].setdefault(video["id"], now.isoformat())
        save_state(state)
    return done


def check_channels() -> None:
    """Una pasada por todos los canales configurados."""
    channels = load_channels()
    if not channels["channels"]:
        log.warning("No hay canales configurados (usa /add en Telegram o YOUTUBE_CHANNEL_ID en .env)")
        return
    state = load_state()
    now = datetime.now(timezone.utc)
    total = 0
    for channel in channels["channels"].values():
        try:
            total += check_one_channel(channel, state, now)
        except Exception as exc:  # noqa: BLE001
            log_error(f"Error comprobando el canal {channel.get('name')}", exc)
    log.info("Pasada terminada: %d vídeo(s) resumido(s)", total)


def seed_channels_from_env() -> None:
    """Primera ejecución: si channels.json no existe, lo creamos con el canal del .env."""
    if CONFIG["channels_file"].exists() or not CONFIG["seed_channel"]:
        return
    log.info("Creando %s a partir de YOUTUBE_CHANNEL_ID=%s", CONFIG["channels_file"], CONFIG["seed_channel"])
    channels = load_channels()
    channel = resolve_channel(CONFIG["seed_channel"])
    channels["channels"][channel["id"]] = channel | {"prompt": None}
    save_channels(channels)


# --------------------------------------------------------------------------- comandos de Telegram

HELP_TEXT = """Envíame una o varias URLs de YouTube y te devuelvo un resumen. Si añades texto junto a la URL, lo uso como instrucciones para ese resumen.

/video <url> [instrucciones] — resume solo ese vídeo (ignora listas de reproducción y otras URLs del mensaje)

Canales vigilados:
/channels — lista los canales y si tienen prompt propio
/add <url o @handle> — vigila un canal nuevo (si la URL acaba en /streams vigila los directos)
/remove <canal> — deja de vigilarlo
/tabs <canal> videos|streams|both — qué pestaña vigilar (vídeos normales, directos o ambas)

Estilo de resumen:
/prompt <canal> — muestra el prompt del canal
/prompt <canal> <texto> — fija el prompt del canal
/prompt <canal> reset — vuelve al prompt por defecto
/default — muestra el prompt por defecto
/default <texto> — cambia el prompt por defecto
/default reset — restaura el prompt original

<canal> puede ser el número de /channels, el @handle o parte del nombre."""


def find_channel(channels: dict, ref: str) -> dict | None:
    entries = list(channels["channels"].values())
    ref = ref.strip()
    if ref.isdigit() and 1 <= int(ref) <= len(entries):
        return entries[int(ref) - 1]
    low = ref.lower().lstrip("@")
    for entry in entries:
        if entry["id"].lower() == low or entry.get("handle", "").lower().lstrip("@") == low:
            return entry
    matches = [e for e in entries if low and low in e["name"].lower()]
    return matches[0] if len(matches) == 1 else None


def format_channels(channels: dict) -> str:
    entries = list(channels["channels"].values())
    if not entries:
        return "No hay canales vigilados. Añade uno con /add <url o @handle>."
    lines = ["Canales vigilados:"]
    for i, entry in enumerate(entries, 1):
        tag = "prompt propio" if entry.get("prompt") else "prompt por defecto"
        lines.append(f"{i}. {entry['name']} ({entry.get('handle') or entry['id']}) — {'+'.join(channel_tabs(entry))}, {tag}")
    lines.append("\nPrompt por defecto: " + ("personalizado" if channels.get("default_prompt") else "original"))
    return "\n".join(lines)


def handle_command(text: str) -> str:
    """Ejecuta un comando de gestión y devuelve el texto de respuesta."""
    parts = text.strip().split(maxsplit=1)
    command = parts[0].lower().split("@")[0]  # "/prompt@MiBot" -> "/prompt"
    arg = parts[1].strip() if len(parts) > 1 else ""
    channels = load_channels()

    if command in ("/start", "/help"):
        return HELP_TEXT

    if command == "/channels":
        return format_channels(channels)

    if command == "/add":
        if not arg:
            return "Uso: /add <url o @handle>"
        try:
            channel = resolve_channel(arg)
        except Exception as exc:  # noqa: BLE001
            return f"No he podido resolver ese canal: {short_error(exc)}"
        if channel["id"] in channels["channels"]:
            return f"{channel['name']} ya estaba en la lista."
        channels["channels"][channel["id"]] = channel | {"prompt": None}
        save_channels(channels)
        return f"✅ Vigilando {channel['name']} ({channel['url']}). Usa /prompt para darle un estilo propio."

    if command == "/remove":
        channel = find_channel(channels, arg) if arg else None
        if not channel:
            return "No encuentro ese canal. Uso: /remove <número, @handle o nombre>\n\n" + format_channels(channels)
        del channels["channels"][channel["id"]]
        save_channels(channels)
        return f"🗑 Ya no vigilo {channel['name']}."

    if command == "/tabs":
        ref, _, choice = arg.partition(" ")
        channel = find_channel(channels, ref) if ref else None
        if not channel:
            return "No encuentro ese canal. Uso: /tabs <canal> videos|streams|both\n\n" + format_channels(channels)
        choice = choice.strip().lower()
        if choice not in ("videos", "streams", "both"):
            return f"{channel['name']} vigila: {'+'.join(channel_tabs(channel))}. Uso: /tabs <canal> videos|streams|both"
        channel["tabs"] = list(CHANNEL_TABS) if choice == "both" else [choice]
        save_channels(channels)
        return f"✅ {channel['name']} ahora vigila: {'+'.join(channel['tabs'])}"

    if command == "/prompt":
        ref, _, new_prompt = arg.partition(" ") if not arg.startswith('"') else (arg, "", "")
        # Permitir el texto en la línea siguiente: "/prompt @canal\ntexto..."
        if "\n" in ref:
            ref, _, rest = ref.partition("\n")
            new_prompt = (rest + "\n" + new_prompt).strip()
        channel = find_channel(channels, ref) if ref else None
        if not channel:
            return "No encuentro ese canal. Uso: /prompt <número, @handle o nombre> [texto | reset]\n\n" + format_channels(channels)
        new_prompt = new_prompt.strip()
        if not new_prompt:
            current = channel.get("prompt")
            return (f"Prompt de {channel['name']}:\n\n{current}" if current
                    else f"{channel['name']} usa el prompt por defecto:\n\n{channels.get('default_prompt') or DEFAULT_INSTRUCTIONS}")
        if new_prompt.lower() == "reset":
            channel["prompt"] = None
            save_channels(channels)
            return f"✅ {channel['name']} vuelve a usar el prompt por defecto."
        channel["prompt"] = new_prompt
        save_channels(channels)
        return f"✅ Prompt de {channel['name']} actualizado:\n\n{new_prompt}"

    if command == "/default":
        if not arg:
            current = channels.get("default_prompt")
            return f"Prompt por defecto ({'personalizado' if current else 'original'}):\n\n{current or DEFAULT_INSTRUCTIONS}"
        if arg.lower() == "reset":
            channels["default_prompt"] = None
            save_channels(channels)
            return "✅ Prompt por defecto restaurado al original."
        channels["default_prompt"] = arg
        save_channels(channels)
        return f"✅ Prompt por defecto actualizado:\n\n{arg}"

    return f"Comando desconocido: {command}\n\n{HELP_TEXT}"


def instructions_from_text(text: str) -> str | None:
    """El texto que acompaña a las URLs son instrucciones puntuales para el resumen."""
    # Quitamos la URL completa, incluidos parámetros como &t=120s o &list=... que van tras el ID
    extra = re.sub(YOUTUBE_URL_RE.pattern + r"\S*", "", text)
    return re.sub(r"[ \t]+", " ", extra).strip() or None


def summarize_requested(chat_id: int, ids: list[str], extra: str | None) -> None:
    for vid in ids:
        try:
            process_video({"id": vid, "title": vid, "url": video_url(vid)}, chat_id, eager=True, extra_instructions=extra)
        except Exception as exc:  # noqa: BLE001
            log_error(f"Error procesando {vid} pedido por Telegram", exc)


def handle_message(chat_id: int, text: str) -> None:
    parts = text.strip().split(maxsplit=1) if text.startswith("/") else [""]
    command = parts[0].lower().split("@")[0]  # "/video@MiBot" -> "/video"
    arg = parts[1] if len(parts) > 1 else ""
    if command == "/video":
        ids = extract_video_ids(arg)
        if not ids:
            send_message(chat_id, "Uso: /video <url de YouTube> [instrucciones opcionales]", preview=False)
            return
        summarize_requested(chat_id, ids[:1], instructions_from_text(arg))
        return

    if text.startswith("/"):
        send_message(chat_id, handle_command(text), preview=False)
        return

    ids = extract_video_ids(text)
    if not ids:
        send_message(chat_id, "No veo ninguna URL de YouTube en el mensaje.\n\n" + HELP_TEXT, preview=False)
        return
    summarize_requested(chat_id, ids, instructions_from_text(text))


def handle_telegram_updates() -> None:
    """Long-polling de Telegram: atiende comandos y URLs del chat autorizado."""
    state = load_state()
    updates = telegram_api(
        "getUpdates",
        offset=state["telegram_offset"],
        timeout=TELEGRAM_LONG_POLL_SECONDS,
        allowed_updates=["message"],
    )
    for upd in updates:
        # Confirmamos el update antes de procesarlo para no repetirlo si algo falla a mitad
        state = load_state()
        state["telegram_offset"] = upd["update_id"] + 1
        save_state(state)

        msg = upd.get("message") or {}
        chat_id = (msg.get("chat") or {}).get("id")
        text = msg.get("text") or msg.get("caption") or ""
        if str(chat_id) != str(CONFIG["telegram_chat_id"]):
            log.warning("Mensaje ignorado de chat no autorizado %s", chat_id)
            continue
        try:
            handle_message(chat_id, text)
        except Exception as exc:  # noqa: BLE001
            log_error(f"Error atendiendo el mensaje {text[:60]!r}", exc)


# --------------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--once", action="store_true", help="Una sola pasada por los canales y termina (para cron)")
    parser.add_argument("--get-chat-id", action="store_true", help="Muestra los chat_id que han escrito al bot")
    parser.add_argument("--video", metavar="URL", help="Resume y envía un vídeo concreto (no toca el estado)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
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

    if args.video:
        ids = extract_video_ids(args.video) or [args.video]
        for vid in ids:
            ok = process_video({"id": vid, "title": vid, "url": video_url(vid)}, CONFIG["telegram_chat_id"], eager=True)
            if not ok:
                sys.exit("El vídeo no tiene subtítulos disponibles todavía")
        return

    seed_channels_from_env()

    if args.once:
        check_channels()
        return

    log.info(
        "Escuchando Telegram y comprobando los canales cada %.0f minutos (Ctrl+C para salir)",
        CONFIG["poll_minutes"],
    )
    next_channel_check = 0.0
    telegram_failures = 0
    while True:
        if time.time() >= next_channel_check:
            next_channel_check = time.time() + CONFIG["poll_minutes"] * 60
            try:
                check_channels()
            except Exception as exc:  # noqa: BLE001
                log_error("Error comprobando los canales; se reintentará en la siguiente pasada", exc)
        try:
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
