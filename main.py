#!/usr/bin/env python3
"""
Vigila un canal de YouTube, descarga los subtítulos de los vídeos nuevos (vía yt-dlp),
los resume con la Inference API de Hetzner (compatible con OpenAI) y envía el resumen
junto con el enlace por Telegram.

Uso:
    python main.py                # bucle: comprueba cada POLL_INTERVAL_MINUTES
    python main.py --once         # una sola pasada (ideal para cron)
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

# --------------------------------------------------------------------------- estado


def load_state(path: Path) -> dict:
    if path.exists():
        with path.open(encoding="utf-8") as fh:
            state = json.load(fh)
    else:
        state = {}
    state.setdefault("processed", {})  # video_id -> info
    state.setdefault("pending", {})    # video_id -> primera vez visto (ISO)
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
        videos.append({
            "id": vid,
            "title": entry.get("title") or vid,
            "url": f"https://www.youtube.com/watch?v={vid}",
        })
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


def fetch_transcript(video_url: str, langs: list[str]) -> tuple[dict, str | None, str]:
    """
    Extrae info del vídeo y su transcripción.
    Devuelve (info, texto|None, descripción de la fuente).
    Prioridad: subtítulos manuales > automáticos, en el orden de `langs`.
    """
    with yt_dlp.YoutubeDL(ydl_base_opts()) as ydl:
        info = ydl.extract_info(video_url, download=False)
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
    )
    text = response.choices[0].message.content or ""
    # Los modelos Qwen "thinking" pueden devolver el razonamiento entre <think>...</think>
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    return text


# --------------------------------------------------------------------------- telegram


def telegram_api(method: str, **payload) -> dict:
    url = f"https://api.telegram.org/bot{CONFIG['telegram_token']}/{method}"
    resp = requests.post(url, json=payload, timeout=30)
    data = resp.json()
    if not data.get("ok"):
        raise RuntimeError(f"Telegram {method} falló: {data}")
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


def send_telegram(text: str) -> None:
    for i, chunk in enumerate(split_message(text)):
        telegram_api(
            "sendMessage",
            chat_id=CONFIG["telegram_chat_id"],
            text=chunk,
            disable_web_page_preview=(i > 0),  # solo el primer trozo muestra la previsualización del vídeo
        )


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


def process_video(video: dict, channel_name: str) -> bool:
    """Descarga subtítulos, resume y envía. Devuelve True si se completó, False si aún no hay subtítulos."""
    log.info("Procesando %s — %s", video["id"], video["title"])
    info, transcript, source = fetch_transcript(video["url"], CONFIG["subtitle_langs"])

    live_status = info.get("live_status")
    if live_status in ("is_live", "is_upcoming"):
        log.info("Vídeo en directo/programado (%s), se reintentará más tarde", live_status)
        return False
    if not transcript:
        log.info("Todavía no hay subtítulos disponibles, se reintentará más tarde")
        return False

    title = info.get("title") or video["title"]
    duration_min = (info.get("duration") or 0) / 60 or None
    log.info("Transcripción obtenida (%s, %d chars). Resumiendo con %s…", source, len(transcript), CONFIG["hetzner_model"])
    summary = summarize(transcript, title, info.get("channel") or channel_name, duration_min)

    message = f"🎬 {title}\n{video['url']}\n\n{summary}"
    send_telegram(message)
    log.info("Resumen enviado por Telegram")
    return True


def run_once() -> None:
    channel_url = normalize_channel_url(CONFIG["channel"])
    state = load_state(CONFIG["state_file"])
    first_run = not CONFIG["state_file"].exists()

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
            ok = process_video(video, CONFIG["channel"])
        except Exception:  # noqa: BLE001
            log.exception("Error procesando %s; se reintentará en la siguiente pasada", video["id"])
            continue

        if ok:
            state["processed"][video["id"]] = {"title": video["title"], "at": now.isoformat()}
            state["pending"].pop(video["id"], None)
            done += 1
        else:
            state["pending"].setdefault(video["id"], now.isoformat())
        save_state(CONFIG["state_file"], state)

    log.info("Pasada terminada: %d vídeo(s) resumido(s)", done)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--once", action="store_true", help="Ejecuta una sola pasada y termina (para cron)")
    parser.add_argument("--get-chat-id", action="store_true", help="Muestra los chat_id que han escrito al bot")
    parser.add_argument("--video", metavar="URL", help="Resume y envía un vídeo concreto (no toca el estado)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    env("TELEGRAM_BOT_API_KEY", required=True)
    if args.get_chat_id:
        print_chat_ids()
        return

    env("HETZNER_INFERENCE_API_KEY", required=True)
    env("TELEGRAM_CHAT_ID", required=True)

    if args.video:
        match = re.search(r"(?:v=|youtu\.be/|shorts/)([\w-]{11})", args.video)
        vid = match.group(1) if match else args.video
        ok = process_video({"id": vid, "title": vid, "url": f"https://www.youtube.com/watch?v={vid}"}, "")
        if not ok:
            sys.exit("El vídeo no tiene subtítulos disponibles todavía")
        return

    env("YOUTUBE_CHANNEL_ID", required=True)

    if args.once:
        run_once()
        return

    log.info("Modo bucle: comprobando cada %.0f minutos (Ctrl+C para salir)", CONFIG["poll_minutes"])
    while True:
        try:
            run_once()
        except KeyboardInterrupt:
            raise
        except Exception:  # noqa: BLE001
            log.exception("Error en la pasada; se reintentará en la siguiente")
        time.sleep(CONFIG["poll_minutes"] * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
