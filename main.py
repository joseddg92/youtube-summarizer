#!/usr/bin/env python3
"""
Vigila un canal de YouTube, descarga los subtítulos de los vídeos nuevos (vía yt-dlp),
los resume con la Inference API de Hetzner (compatible con OpenAI) y envía el resumen
junto con el enlace por Telegram. Además, escucha el bot de Telegram: cualquier URL de
YouTube que le envíes se resume al momento.

Uso:
    python main.py                # bucle: escucha Telegram y comprueba el canal cada POLL_INTERVAL_MINUTES
    python main.py --once         # una sola pasada por el canal (ideal para cron; no escucha Telegram)
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
    "channel": env("YOUTUBE_CHANNEL_ID"),
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
}

# Límite de seguridad para la transcripción (~100k tokens; el modelo admite 262k)
MAX_TRANSCRIPT_CHARS = 400_000
TELEGRAM_MAX_LEN = 4000  # el límite real es 4096
TELEGRAM_LONG_POLL_SECONDS = 25

YOUTUBE_URL_RE = re.compile(
    r"(?:https?://)?(?:www\.|m\.)?"
    r"(?:youtube\.com/(?:watch\?(?:[^\s]*&)?v=|shorts/|live/|embed/)|youtu\.be/)"
    r"([\w-]{11})"
)

# --------------------------------------------------------------------------- estado


def load_state(path: Path) -> dict:
    if path.exists():
        with path.open(encoding="utf-8") as fh:
            state = json.load(fh)
    else:
        state = {}
    state.setdefault("processed", {})  # video_id -> info
    state.setdefault("pending", {})    # video_id -> primera vez visto (ISO)
    state.setdefault("telegram_offset", 0)
    return state


def save_state(path: Path, state: dict) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(state, fh, ensure_ascii=False, indent=2)
    tmp.replace(path)


# --------------------------------------------------------------------------- youtube


def normalize_channel_url(channel: str) -> str:
    """Convierte URL / @handle / UC... en la URL de la pestaña de vídeos del canal."""
    channel = channel.strip()
    if channel.startswith("http"):
        url = channel.rstrip("/")
    elif channel.startswith("@"):
        url = f"https://www.youtube.com/{channel}"
    elif channel.startswith("UC"):
        url = f"https://www.youtube.com/channel/{channel}"
    else:
        url = f"https://www.youtube.com/@{channel}"
    # Si ya apunta a una pestaña concreta (videos, streams, shorts...) la respetamos
    if not re.search(r"/(videos|streams|shorts|live)$", url):
        url += "/videos"
    return url


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


def list_latest_videos(channel_url: str, limit: int) -> list[dict]:
    """Devuelve los últimos `limit` vídeos del canal (más recientes primero) sin descargar nada."""
    opts = ydl_base_opts() | {"extract_flat": "in_playlist", "playlistend": limit}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(channel_url, download=False)
    videos = []
    for entry in info.get("entries") or []:
        vid = entry.get("id")
        if not vid:
            continue
        videos.append({"id": vid, "title": entry.get("title") or vid, "url": video_url(vid)})
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


def fetch_transcript(url: str, langs: list[str]) -> tuple[dict, str | None, str]:
    """
    Extrae info del vídeo y su transcripción.
    Devuelve (info, texto|None, descripción de la fuente).
    Prioridad: subtítulos manuales > automáticos, en el orden de `langs`.
    """
    with yt_dlp.YoutubeDL(ydl_base_opts()) as ydl:
        info = ydl.extract_info(url, download=False)
        sources = (
            ("manual", info.get("subtitles") or {}),
            ("auto", info.get("automatic_captions") or {}),
        )
        for kind, table in sources:
            for lang in langs:
                for key, formats in table.items():
                    if key != lang and not key.startswith(lang + "-"):
                        continue
                    fmt = next((f for f in formats if f.get("ext") == "json3" and f.get("url")), None)
                    if not fmt:
                        continue
                    try:
                        raw = ydl.urlopen(fmt["url"]).read()
                        text = parse_json3(raw)
                    except Exception as exc:  # noqa: BLE001
                        log.warning("No se pudo descargar subtítulo %s/%s: %s", kind, key, exc)
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


def summarize(transcript: str, title: str, channel: str, duration_min: float | None) -> str:
    if len(transcript) > MAX_TRANSCRIPT_CHARS:
        log.warning("Transcripción muy larga (%d chars), se recorta", len(transcript))
        transcript = transcript[:MAX_TRANSCRIPT_CHARS]

    client = OpenAI(base_url=CONFIG["hetzner_base_url"], api_key=CONFIG["hetzner_api_key"])

    system_prompt = (
        f"Eres un asistente que resume vídeos de YouTube a partir de su transcripción. "
        f"Responde siempre en {CONFIG['summary_language']}. "
        "El resultado se enviará por Telegram como texto plano: NO uses Markdown "
        "(nada de **, #, ``` ni tablas). Usa el guion '-' para las listas.\n\n"
        "Estructura:\n"
        "1. Un párrafo de 2-3 frases con la idea principal.\n"
        "2. 'Puntos clave:' seguido de 4-8 viñetas concretas (datos, cifras, argumentos, recomendaciones).\n"
        "3. 'Conclusión:' una frase con la postura o recomendación final del autor.\n"
        "Sé fiel al contenido, no inventes nada y no añadas opiniones propias. Máximo ~300 palabras."
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
        max_tokens=1500,
        # Qwen3 es un modelo "thinking": sin esto gasta todos los tokens razonando y devuelve content vacío
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    choice = response.choices[0]
    text = clean_summary(choice.message.content or "")
    if not text:
        raise RuntimeError(f"El modelo devolvió una respuesta vacía (finish_reason={choice.finish_reason})")
    return text


# --------------------------------------------------------------------------- telegram


def telegram_api(method: str, **payload) -> dict:
    url = f"https://api.telegram.org/bot{CONFIG['telegram_token']}/{method}"
    resp = requests.post(url, json=payload, timeout=TELEGRAM_LONG_POLL_SECONDS + 15)
    data = resp.json()
    if not data.get("ok"):
        raise RuntimeError(f"Telegram {method} falló: {data.get('description', data)}")
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

    def _render(self, chat_id, text: str, preview: bool) -> None:
        options = {"is_disabled": not preview, "prefer_small_media": True}
        if self.message_id is None:
            result = telegram_api("sendMessage", chat_id=chat_id, text=text, link_preview_options=options)
            self.message_id = result["message_id"]
            return
        try:
            telegram_api("editMessageText", chat_id=chat_id, message_id=self.message_id,
                         text=text, link_preview_options=options)
        except RuntimeError as exc:
            if "message is not modified" not in str(exc):
                raise

    def update(self, status: str) -> None:
        # Sin previsualización mientras trabaja, para que el mensaje se vea compacto
        self._render(self.chat_id, f"{self._header()}\n\n⏳ {status}", preview=False)

    def finish(self, summary: str) -> None:
        chunks = split_message(f"{self._header()}\n\n{summary}")
        self._render(self.chat_id, chunks[0], preview=True)
        for chunk in chunks[1:]:
            send_message(self.chat_id, chunk, preview=False)

    def fail(self, reason: str) -> None:
        self._render(self.chat_id, f"{self._header()}\n\n❌ {reason}", preview=False)


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


def process_video(video: dict, channel_name: str, chat_id: str | int, eager: bool) -> bool:
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
            progress.fail(f"No se pudo acceder al vídeo: {str(exc).splitlines()[0][:200]}")
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

    duration_min = (info.get("duration") or 0) / 60 or None
    log.info("Transcripción obtenida (%s, %d chars). Resumiendo con %s…", source, len(transcript), CONFIG["hetzner_model"])
    details = f"{duration_min:.0f} min, " if duration_min else ""
    progress.update(f"Resumiendo ({details}{len(transcript) // 1000}k caracteres de transcripción)…")

    try:
        summary = summarize(transcript, progress.title, info.get("channel") or channel_name, duration_min)
    except Exception as exc:  # noqa: BLE001
        progress.fail(f"Error al resumir: {str(exc)[:200]}")
        raise

    progress.finish(summary)
    log.info("Resumen enviado por Telegram")
    return True


def check_channel() -> None:
    """Una pasada por el canal: resume los vídeos nuevos."""
    channel_url = normalize_channel_url(CONFIG["channel"])
    state = load_state(CONFIG["state_file"])
    first_run = not state["processed"]

    videos = list_latest_videos(channel_url, CONFIG["check_latest_n"])
    if not videos:
        log.warning("No se encontraron vídeos en %s", channel_url)
        return
    log.info("%d vídeos recientes en el canal; %d ya procesados", len(videos), len(state["processed"]))

    now = datetime.now(timezone.utc)

    if first_run:
        # Marcamos como vistos todos menos los N más recientes para no reventar el chat con vídeos antiguos
        for v in videos[CONFIG["first_run_videos"]:]:
            state["processed"][v["id"]] = {"title": v["title"], "skipped_on_first_run": True, "at": now.isoformat()}
        save_state(CONFIG["state_file"], state)

    new_videos = [v for v in videos if v["id"] not in state["processed"]]
    # Los más antiguos primero, para que lleguen por Telegram en orden cronológico
    new_videos.reverse()

    done = 0
    for video in new_videos:
        if done >= CONFIG["max_per_run"]:
            log.info("Alcanzado MAX_VIDEOS_PER_RUN, el resto quedará para la siguiente pasada")
            break

        first_seen = state["pending"].get(video["id"])
        if first_seen:
            age = now - datetime.fromisoformat(first_seen)
            if age > timedelta(hours=CONFIG["give_up_hours"]):
                log.warning("Descartando %s: sin subtítulos tras %.0f h", video["id"], age.total_seconds() / 3600)
                state["processed"][video["id"]] = {"title": video["title"], "gave_up": True, "at": now.isoformat()}
                state["pending"].pop(video["id"], None)
                save_state(CONFIG["state_file"], state)
                continue

        try:
            ok = process_video(video, CONFIG["channel"], CONFIG["telegram_chat_id"], eager=False)
        except Exception:  # noqa: BLE001
            log.exception("Error procesando %s; se reintentará en la siguiente pasada", video["id"])
            continue

        # Recargamos por si handle_telegram_updates ha tocado el fichero mientras tanto
        state = load_state(CONFIG["state_file"])
        if ok:
            state["processed"][video["id"]] = {"title": video["title"], "at": now.isoformat()}
            state["pending"].pop(video["id"], None)
            done += 1
        else:
            state["pending"].setdefault(video["id"], now.isoformat())
        save_state(CONFIG["state_file"], state)

    log.info("Pasada terminada: %d vídeo(s) resumido(s)", done)


HELP_TEXT = (
    "Envíame una o varias URLs de YouTube y te devuelvo un resumen del vídeo.\n"
    "Además vigilo el canal configurado y te aviso de los vídeos nuevos."
)


def handle_telegram_updates() -> None:
    """Long-polling de Telegram: resume cualquier URL de YouTube que llegue al chat autorizado."""
    state = load_state(CONFIG["state_file"])
    updates = telegram_api(
        "getUpdates",
        offset=state["telegram_offset"],
        timeout=TELEGRAM_LONG_POLL_SECONDS,
        allowed_updates=["message"],
    )
    for upd in updates:
        # Confirmamos el update antes de procesarlo para no repetirlo si algo falla a mitad
        state = load_state(CONFIG["state_file"])
        state["telegram_offset"] = upd["update_id"] + 1
        save_state(CONFIG["state_file"], state)

        msg = upd.get("message") or {}
        chat_id = (msg.get("chat") or {}).get("id")
        text = msg.get("text") or msg.get("caption") or ""
        if str(chat_id) != str(CONFIG["telegram_chat_id"]):
            log.warning("Mensaje ignorado de chat no autorizado %s", chat_id)
            continue

        ids = extract_video_ids(text)
        if not ids:
            if text.startswith("/"):
                send_message(chat_id, HELP_TEXT, preview=False)
            else:
                send_message(chat_id, "No veo ninguna URL de YouTube en el mensaje.\n\n" + HELP_TEXT, preview=False)
            continue

        for vid in ids:
            try:
                process_video({"id": vid, "title": vid, "url": video_url(vid)}, "", chat_id, eager=True)
            except Exception:  # noqa: BLE001
                log.exception("Error procesando %s pedido por Telegram", vid)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--once", action="store_true", help="Una sola pasada por el canal y termina (para cron)")
    parser.add_argument("--get-chat-id", action="store_true", help="Muestra los chat_id que han escrito al bot")
    parser.add_argument("--video", metavar="URL", help="Resume y envía un vídeo concreto (no toca el estado)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    env("TELEGRAM_BOT_API_KEY", required=True)
    if args.get_chat_id:
        print_chat_ids()
        return

    env("HETZNER_INFERENCE_API_KEY", required=True)
    env("TELEGRAM_CHAT_ID", required=True)

    if args.video:
        ids = extract_video_ids(args.video) or [args.video]
        for vid in ids:
            ok = process_video({"id": vid, "title": vid, "url": video_url(vid)}, "", CONFIG["telegram_chat_id"], eager=True)
            if not ok:
                sys.exit("El vídeo no tiene subtítulos disponibles todavía")
        return

    env("YOUTUBE_CHANNEL_ID", required=True)

    if args.once:
        check_channel()
        return

    log.info(
        "Escuchando Telegram y comprobando el canal cada %.0f minutos (Ctrl+C para salir)",
        CONFIG["poll_minutes"],
    )
    next_channel_check = 0.0
    while True:
        if time.time() >= next_channel_check:
            next_channel_check = time.time() + CONFIG["poll_minutes"] * 60
            try:
                check_channel()
            except Exception:  # noqa: BLE001
                log.exception("Error comprobando el canal; se reintentará en la siguiente pasada")
        try:
            handle_telegram_updates()  # bloquea hasta TELEGRAM_LONG_POLL_SECONDS si no hay mensajes
        except Exception:  # noqa: BLE001
            log.exception("Error atendiendo Telegram")
            time.sleep(5)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
