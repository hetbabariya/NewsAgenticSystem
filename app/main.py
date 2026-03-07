from fastapi import FastAPI

from app.telegram.router import router as telegram_router

app = FastAPI(title="News Agentic System")

app.include_router(telegram_router)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
