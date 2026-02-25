from fastapi import FastAPI

app = FastAPI(title="MTG Invasion Recognizer API")


@app.get("/api/v1/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}
