import gradio as gr
from app.main import app as fastapi_app

# Mounts your FastAPI app into Gradio to run on Hugging Face's free 16GB RAM server
app = gr.mount_gradio_app(
    fastapi_app, 
    gr.Interface(lambda x: x, "text", "text"), 
    path="/ui"
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)