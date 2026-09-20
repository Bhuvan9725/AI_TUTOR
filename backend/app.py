from typing import List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.model_service import TutorModel


app = FastAPI(
    title="Adaptive Multilingual AI Tutor",
    version="2.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


tutor = TutorModel()


class Message(BaseModel):

    role: str

    content: str


class ChatRequest(BaseModel):

    message: str

    language: str = "english"

    level: str = "intermediate"

    mode: str = "tutor"

    history: Optional[
        List[Message]
    ] = None


class ChatResponse(BaseModel):

    answer: str

    language: str

    level: str

    mode: str


@app.get("/")
def home():

    return {
        "message":
            "Adaptive Multilingual AI Tutor API",
        "status":
            "running"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


@app.post(
    "/chat",
    response_model=ChatResponse
)
def chat(request: ChatRequest):

    history = None

    if request.history:

        history = [
            {
                "role": item.role,
                "content": item.content
            }

            for item in request.history
        ]

    answer = tutor.generate(
        message=request.message,
        language=request.language,
        level=request.level,
        mode=request.mode,
        history=history
    )

    return ChatResponse(
        answer=answer,
        language=request.language,
        level=request.level,
        mode=request.mode
    )