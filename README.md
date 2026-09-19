# youtube-summaricer

Vigila un canal de YouTube, descarga los subtítulos de cada vídeo nuevo con **yt-dlp**,
los resume con la **Inference API de Hetzner** (compatible con OpenAI) y te manda el
resumen + enlace por **Telegram**. También puedes enviarle al bot cualquier URL de YouTube
y te la resume al momento, editando un mensaje de progreso mientras trabaja.

## Instalación

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # y rellena las claves
```

## Configuración (`.env`)

| Variable | Descripción |
|---|---|
| `HETZNER_INFERENCE_API_KEY` | Token creado en https://experiments.hetzner.com/ |
| `HETZNER_INFERENCE_MODEL` | `Qwen/Qwen3.6-35B-A3B-FP8` (por defecto) o `Qwen3.8-27B` |
| `YOUTUBE_CHANNEL_ID` | URL, `@handle` o ID `UC…` del canal (ej. `https://www.youtube.com/@MeetKevin`) |
| `TELEGRAM_BOT_API_KEY` | Token que te da [@BotFather](https://t.me/BotFather) |
| `TELEGRAM_CHAT_ID` | Chat destino. Escribe algo a tu bot y ejecuta `python main.py --get-chat-id` |
| `YTDLP_COOKIES_FROM_BROWSER` | Opcional. Solo si YouTube devuelve *"Sign in to confirm you're not a bot"* (pasa tras muchas peticiones seguidas o desde IPs de datacenter): navegador del que leer cookies (`firefox`, `chrome`, `safari`, `brave`…) |
| `SUBTITLE_LANGUAGES` | Idiomas de subtítulos a intentar, en orden (`en,es`) |
| `SUMMARY_LANGUAGE` | Idioma del resumen (`español`) |

El resto de variables están comentadas en [.env.example](.env.example).

## Uso

```bash
python main.py                   # modo bot: escucha Telegram + comprueba el canal cada POLL_INTERVAL_MINUTES
python main.py --once            # una pasada por el canal y sale (para cron / launchd; no escucha Telegram)
python main.py --video <URL>     # prueba con un vídeo concreto
```

En modo bot, cualquier mensaje con URLs de YouTube que envíes al chat configurado
(`TELEGRAM_CHAT_ID`) se resume al instante; mensajes de otros chats se ignoran.

En la primera ejecución solo se resumen los `FIRST_RUN_VIDEOS` vídeos más recientes; el
resto se marcan como vistos en `state.json` para no inundar el chat.

Ejemplo de cron (cada 30 min):

```
*/30 * * * * cd /ruta/youtube-summaricer && .venv/bin/python main.py --once >> summarizer.log 2>&1
```

## Cómo funciona

1. `yt-dlp` lista los últimos `CHECK_LATEST_N` vídeos de la pestaña *Videos* del canal (sin descargar nada).
2. Para cada vídeo no visto, obtiene los subtítulos (manuales > automáticos) en formato `json3` y los convierte a texto plano. Si el vídeo es un directo o aún no tiene subtítulos, se reintenta en la siguiente pasada (hasta `GIVE_UP_AFTER_HOURS`).
3. Envía la transcripción a `POST {HETZNER_INFERENCE_BASE_URL}/chat/completions` con el SDK de OpenAI (con `enable_thinking: false`; si no, Qwen gasta todos los tokens razonando y devuelve una respuesta vacía).
4. Manda `🎬 título + enlace + resumen` por Telegram (troceado si supera 4096 caracteres). Mientras trabaja, edita un mensaje con el progreso ("Obteniendo subtítulos…", "Resumiendo…").

Límites de la API de Hetzner (por token): 10 peticiones/min, 4M tokens de entrada/min. `MAX_VIDEOS_PER_RUN` evita superarlos.
