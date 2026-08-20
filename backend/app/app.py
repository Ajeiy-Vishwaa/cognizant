import sys
from pathlib import Path

# Add the directory containing 'app' to system path
sys.path.append(str(Path(__file__).resolve().parent))

import gradio as gr
from app.main import app as fastapi_app

# Mount FastAPI inside Gradio
app = gr.mount_gradio_app(
    fastapi_app, 
    gr.Interface(lambda x: x, "text", "text"), 
    path="/ui"
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)