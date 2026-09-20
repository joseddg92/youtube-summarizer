# youtube-summaricer

Vigila uno o varios canales de YouTube, descarga los subtítulos de cada vídeo nuevo con
**yt-dlp**, los resume con la **Inference API de Hetzner** (compatible con OpenAI) y te manda
el resumen + enlace por **Telegram**. También puedes enviarle al bot cualquier URL de YouTube
y te la resume al momento, editando un mensaje de progreso mientras trabaja. Cada canal puede
tener su propio estilo de resumen (por ejemplo, "enfocado a un trader"), gestionable desde Telegram.

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
| `YOUTUBE_CHANNEL_ID` | Canal inicial (URL, `@handle` o ID `UC…`). Solo se usa para crear `channels.json` la primera vez |
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
(`TELEGRAM_CHAT_ID`) se resume al instante; mensajes de otros chats se ignoran. Si acompañas
la URL con texto ("…/watch?v=xxx céntrate en lo que dice de NVIDIA"), ese texto se usa como
instrucciones puntuales para ese resumen.

## Canales y estilos de resumen

Los canales vigilados y su prompt viven en `channels.json` (ver
[channels.example.json](channels.example.json)). Se gestionan desde Telegram:

| Comando | Qué hace |
|---|---|
| `/channels` | Lista los canales y si tienen prompt propio |
| `/add <url o @handle>` | Vigila un canal nuevo. Si la URL acaba en `/streams`, vigila los directos en vez de los vídeos |
| `/remove <canal>` | Deja de vigilarlo |
| `/tabs <canal> videos\|streams\|both` | Qué pestaña del canal vigilar (vídeos normales, directos o ambas) |
| `/prompt <canal>` | Muestra el prompt del canal |
| `/prompt <canal> <texto>` | Fija el prompt del canal (el texto puede ir en la línea siguiente) |
| `/prompt <canal> reset` | Vuelve al prompt por defecto |
| `/default [texto \| reset]` | Muestra / cambia / restaura el prompt por defecto |

`<canal>` puede ser el número que sale en `/channels`, el `@handle` o parte del nombre.

El prompt de un canal se aplica tanto a los vídeos nuevos detectados como a cualquier URL suya
que envíes a mano (se identifica por el `channel_id` del vídeo). Los vídeos resumidos a mano
(`--video` o URL por Telegram) se marcan como procesados para que el vigilante no los repita. El prompt describe solo el
enfoque y la estructura; el idioma, el formato de texto plano y la fidelidad al contenido se
imponen siempre.

En la primera ejecución solo se resumen los `FIRST_RUN_VIDEOS` vídeos más recientes; el
resto se marcan como vistos en `state.json` para no inundar el chat.

Ejemplo de cron (cada 30 min):

```
*/30 * * * * cd /ruta/youtube-summaricer && .venv/bin/python main.py --once >> summarizer.log 2>&1
```

## Cómo funciona

1. `yt-dlp` lista los últimos `CHECK_LATEST_N` vídeos de la pestaña *Videos* del canal (sin descargar nada).
2. Para cada vídeo no visto, obtiene los subtítulos en formato `json3` y los convierte a texto plano. Prioridad: manuales en `SUBTITLE_LANGUAGES` > automáticos en el idioma original del vídeo (`xx-orig`, mejor que las traducciones automáticas) > automáticos en `SUBTITLE_LANGUAGES`. Si el vídeo está en directo o aún no tiene subtítulos (los directos recién terminados tardan horas en tenerlos), se reintenta en cada pasada hasta `GIVE_UP_AFTER_HOURS`.
3. Envía la transcripción a `POST {HETZNER_INFERENCE_BASE_URL}/chat/completions` con el SDK de OpenAI (con `enable_thinking: false`; si no, Qwen gasta todos los tokens razonando y devuelve una respuesta vacía).
4. Manda `🎬 título + enlace + resumen` por Telegram (troceado si supera 4096 caracteres). Mientras trabaja, edita un mensaje con el progreso ("Obteniendo subtítulos…", "Resumiendo…").

Límites de la API de Hetzner (por token): 10 peticiones/min, 4M tokens de entrada/min. `MAX_VIDEOS_PER_RUN` evita superarlos.
