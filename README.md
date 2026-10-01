# youtube-summaricer

Bot de Telegram que resume vídeos de YouTube. Descarga los subtítulos con **yt-dlp**, los resume
con un LLM compatible con OpenAI (por defecto la **Inference API de Hetzner**) y envía el resumen
+ enlace por **Telegram**.

Lo puede usar **cualquier usuario, grupo o canal**: cada chat tiene sus propias suscripciones a
canales de YouTube, cada una con su propio estilo de resumen (prompt). Todas las acciones se
notifican al chat de administración (`TELEGRAM_CHAT_ID`).

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
| `TELEGRAM_BOT_API_KEY` | Token que te da [@BotFather](https://t.me/BotFather) |
| `TELEGRAM_CHAT_ID` | Chat de administración, donde se notifica todo lo que hace cualquiera con el bot. Puede ser un chat privado, un grupo o un canal (el bot debe ser admin). Para obtener el id: `python main.py --get-chat-id` |
| `YTDLP_COOKIES_FROM_BROWSER` | Opcional. Solo si YouTube devuelve *"Sign in to confirm you're not a bot"*: navegador del que leer cookies (`firefox`, `chrome`, `safari`, `brave`…) |
| `SUBTITLE_LANGUAGES` | Idiomas de subtítulos a intentar si no hay del idioma original (`en,es`) |
| `SUMMARY_LANGUAGE` | Idioma del resumen (`español`) |
| `WORKERS` | Resúmenes en paralelo (3) |

El resto de variables están comentadas en [.env.example](.env.example).

## Uso

```bash
python main.py                   # bot: escucha Telegram y comprueba los canales cada POLL_INTERVAL_MINUTES
python main.py --once            # una pasada por los canales suscritos y sale (cron; no escucha Telegram)
python main.py --video <URL>     # resume un vídeo y lo envía al chat de administración
```

Al arrancar, el bot registra sus comandos en Telegram para que aparezcan en el menú `/`.

## Comandos

| Comando | Qué hace |
|---|---|
| `/ayuda`, `/start`, `/help` | Lista de comandos |
| `/video <url>` | Resume ese vídeo. Pregunta cómo resumirlo (texto libre) con botón **Usar prompt por defecto**. Si escribes las instrucciones tras la URL, se salta la pregunta |
| `/suscribirse <url>` | Suscribe el chat a un canal de YouTube. Pregunta el prompt (con botón por defecto) y después ofrece **Resumir los últimos 3** vídeos como ejemplo u **Omitir**; si no se contesta en 30 min, la pregunta se borra y se omite. Con una URL acabada en `/streams` sigue los directos. También acepta la URL de un vídeo del canal |
| `/suscripciones` | Lista las suscripciones del chat con botones para **editar el prompt** o **borrarlas** |

Detalles:

- **Prompt por defecto** en `/video`: si el chat está suscrito al canal del vídeo, se usa el prompt de esa suscripción; si no, el genérico.
- En **privado** basta con escribir el prompt; pegar una URL suelta equivale a `/video`.
- En **grupos** hay que *responder* al mensaje del bot (con el modo privacidad, Telegram no le entrega el resto de mensajes), y solo quien lanzó el comando puede contestar o pulsar los botones.
- En **canales** el bot debe ser administrador; el siguiente post es el prompt, y solo los administradores pueden pulsar los botones (los suscriptores del canal también los ven).
- Si lanzas un comando nuevo con una pregunta pendiente, la anterior se cancela. Las preguntas de prompt caducan a las 24 h.
- Si expulsan o bloquean al bot, se borran las suscripciones de ese chat.

## Datos

- `subscriptions.json`: suscripciones por chat (ver [subscriptions.example.json](subscriptions.example.json)).
- `state.json`: vídeos ya vistos/pendientes, a qué chats se ha enviado cada vídeo y preguntas abiertas.

Si existe un `channels.json` de la versión anterior (un solo usuario) y no hay `subscriptions.json`,
se migra automáticamente al chat `TELEGRAM_CHAT_ID`.

## Cómo funciona

1. Cada `POLL_INTERVAL_MINUTES`, `yt-dlp` lista los últimos `CHECK_LATEST_N` vídeos de cada canal con al menos un suscriptor (pestaña *Videos* y/o *Directos*).
2. Para cada vídeo no visto, obtiene los subtítulos en formato `json3` y los convierte a texto plano. Prioridad: manuales > automáticos del audio original (`xx-orig` del idioma original) > traducciones automáticas. YouTube dobla muchos vídeos con voces de IA y cada doblaje trae su propio `xx-orig`; se descartan detectando cuál es la pista de audio original. Si el vídeo está en directo o aún no tiene subtítulos (los directos tardan horas en tenerlos), se reintenta en cada pasada hasta `GIVE_UP_AFTER_HOURS`.
3. La transcripción se descarga una vez y se resume para cada suscriptor con su prompt (una sola llamada al LLM por prompt distinto), con el SDK de OpenAI (`enable_thinking: false`; si no, Qwen gasta todos los tokens razonando y devuelve una respuesta vacía).
4. Se envía `🎬 título + enlace + resumen` (troceado si supera 4096 caracteres). En las peticiones manuales, el mensaje se va editando con el progreso.
5. Los resúmenes se hacen en un pool de `WORKERS` hilos, así el bot sigue respondiendo mientras trabaja.

Al suscribirse a un canal que nadie seguía, lo ya publicado se marca como visto (para eso están los
ejemplos). Límites de la API de Hetzner (por token): 10 peticiones/min, 4M tokens de entrada/min;
`MAX_VIDEOS_PER_RUN` y `WORKERS` evitan superarlos.
