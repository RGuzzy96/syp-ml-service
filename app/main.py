from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from app.consumers import start_consumer
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up...")
    start_consumer()

    yield

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    return {"message": "Data processing backend is running"}