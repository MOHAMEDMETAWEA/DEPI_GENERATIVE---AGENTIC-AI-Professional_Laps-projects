from fastapi import FastAPI

app = FastAPI(
    title="FastAPI on Colab",
    description="Demo API running on Google Colab with ngrok",
    version="1.0.0"
)

@app.get("/")
def home():
    return {"message": "FastAPI is running on Colab 🚀"}

@app.get("/health")
def health():
    return {"status": "ok"}
