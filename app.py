import os
import time
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from inference import load_model, predict_video
from video_preprocess import PreprocessConfig

APP_DIR = Path(__file__).resolve().parent
UI_DIR = APP_DIR / "ui"
DIST_DIR = UI_DIR / "dist"

app = FastAPI(title="LumiSign Sign Inference")

MODEL = None
LABEL_MAP = None
DEFAULT_LOCAL_CHECKPOINT = APP_DIR / "transformer_large.pth"
DEFAULT_MODEL_CHECKPOINT = (
    str(DEFAULT_LOCAL_CHECKPOINT) if DEFAULT_LOCAL_CHECKPOINT.is_file() else None
)
MODEL_DATASET = os.getenv("MODEL_DATASET", "isl_split_dataset")
MODEL_TYPE = os.getenv("MODEL_TYPE", "transformer")
MODEL_TRANSFORMER_SIZE = os.getenv(
    "MODEL_TRANSFORMER_SIZE",
    "large" if DEFAULT_MODEL_CHECKPOINT else "small",
)
MODEL_CHECKPOINT = os.getenv("MODEL_CHECKPOINT", DEFAULT_MODEL_CHECKPOINT)
MODEL_MAX_FRAME_LEN = int(os.getenv("MODEL_MAX_FRAME_LEN", "169"))

DEFAULT_LABEL_MAP_CANDIDATES = [
    APP_DIR / "label_maps" / f"label_map_{MODEL_DATASET}.json",
    APP_DIR / "label_maps" / "label_map_isl_split_dataset.json",
    APP_DIR / "label_maps" / "label_map_islsplit.json",
]
MODEL_LABEL_MAP_PATH = os.getenv("MODEL_LABEL_MAP_PATH")
if not MODEL_LABEL_MAP_PATH:
    for candidate in DEFAULT_LABEL_MAP_CANDIDATES:
        if candidate.is_file():
            MODEL_LABEL_MAP_PATH = str(candidate)
            break


@app.on_event("startup")
def _load():
    global MODEL, LABEL_MAP
    MODEL, LABEL_MAP = load_model(
        dataset=MODEL_DATASET,
        model_type=MODEL_TYPE,
        transformer_size=MODEL_TRANSFORMER_SIZE,
        checkpoint_path=MODEL_CHECKPOINT,
        label_map_path=MODEL_LABEL_MAP_PATH,
    )


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if MODEL is None or LABEL_MAP is None:
        return JSONResponse({"error": "Model not loaded"}, status_code=500)

    suffix = Path(file.filename).suffix if file.filename else ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    preprocess_config = PreprocessConfig(
        apply_darken=False,
        apply_brighten=True,
        darken_min=0.3,
        darken_max=0.8,
        brighten_method="clahe",
        brighten_gamma_min=1.2,
        brighten_gamma_max=1.8,
    )

    start = time.time()
    try:
        result = predict_video(
            tmp_path,
            dataset=MODEL_DATASET,
            model=MODEL,
            label_map=LABEL_MAP,
            preprocess_config=preprocess_config,
            max_frame_len=MODEL_MAX_FRAME_LEN,
        )
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    elapsed_ms = int((time.time() - start) * 1000)
    result["elapsed_ms"] = elapsed_ms
    return JSONResponse(result)


def _ui_built() -> bool:
    index_path = DIST_DIR / "index.html"
    if not index_path.is_file():
        return False

    src_dir = UI_DIR / "src"
    if not src_dir.is_dir():
        return True

    latest_src_mtime = 0.0
    for root, _, files in os.walk(src_dir):
        for name in files:
            try:
                latest_src_mtime = max(
                    latest_src_mtime,
                    (Path(root) / name).stat().st_mtime,
                )
            except OSError:
                continue

    try:
        return index_path.stat().st_mtime >= latest_src_mtime
    except OSError:
        return True


def _configure_ui(app: FastAPI) -> None:
    if _ui_built():
        app.mount("/", StaticFiles(directory=DIST_DIR, html=True), name="ui")
        return

    @app.get("/")
    def index():
        ui_path = str(UI_DIR)
        html = f"""
        <!doctype html>
        <html lang="en">
          <head>
            <meta charset="utf-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1" />
            <title>UI Not Built</title>
            <style>
              body {{ font-family: Arial, sans-serif; padding: 32px; background: #f8fafc; color: #0f172a; }}
              pre {{ background: #0f172a; color: #e2e8f0; padding: 12px; border-radius: 8px; }}
              code {{ font-family: monospace; }}
            </style>
          </head>
          <body>
            <h1>UI not built yet</h1>
            <p>To run the React UI in development:</p>
            <pre>cd {ui_path}
npm install
npm run dev</pre>
            <p>Then open <code>http://localhost:5173</code> (it proxies <code>/predict</code>).</p>
            <p>To serve the built UI from FastAPI:</p>
            <pre>cd {ui_path}
npm run build</pre>
            <p>Restart the backend after building.</p>
          </body>
        </html>
        """
        return HTMLResponse(html)


_configure_ui(app)
