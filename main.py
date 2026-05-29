from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from agent import root_agent
import dotenv
import uuid
import os
import time
import logging

#  Setup 
dotenv.load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=" AI Agent",
    description="A smart GenAI agent powered by Google ADK and Gemini with tools like search, calculator, and more.",
    version="2.0.0"
)

#  CORS (allows frontend to call this API) 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

#  Constants 
APP_NAME = "tools-agent"
USER_ID  = "bachi"

INITIAL_STATE = {
    "name": "bachi",
    "data": """
    I am a software enginerr with 3 years of experience in the tech industry. 
    I have worked on various projects including web development, data analysis,
    and machine learning. I am passionate about coding and always eager to learn
    new technologies. I love cricket and my favourite player is Rohit Sharma.
    """
}

#  Request / Response Models 
class ChatRequest(BaseModel):
    query: str

    @validator("query")
    def query_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("Query cannot be empty")
        if len(v) > 1000:
            raise ValueError("Query too long (max 1000 characters)")
        return v.strip()

class ChatResponse(BaseModel):
    query: str
    response: str
    session_id: str
    time_taken_seconds: float


#  Core Agent Runner 
async def run_agent(query: str) -> dict:
    session_id = str(uuid.uuid4())
    start_time = time.time()

    session_service = InMemorySessionService()

    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=session_id,
        state=INITIAL_STATE
    )

    runner = Runner(
        agent=root_agent,
        session_service=session_service,
        app_name=APP_NAME
    )

    message = types.Content(
        role="user",
        parts=[types.Part(text=query)]
    )

    final_response = ""
    for event in runner.run(
        user_id=USER_ID,
        session_id=session_id,
        new_message=message,
    ):
        if event.is_final_response():
            if event.content and event.content.parts:
                final_response = event.content.parts[0].text

    elapsed = round(time.time() - start_time, 2)
    logger.info(f"Query: '{query}' | Time: {elapsed}s | Session: {session_id}")

    return {
        "query": query,
        "response": final_response,
        "session_id": session_id,
        "time_taken_seconds": elapsed
    }


#  Endpoints 

@app.get("/", tags=["Health"])
def home():
    """Health check endpoint."""
    return {
        "message": "AI Agent is running ",
        "version": "2.0.0",
        "tools": ["google_search", "calculator", "get_current_datetime", "summarize_text"],
        "endpoints": {
            "GET  /chat":  "Chat via query param",
            "POST /chat":  "Chat via JSON body",
            "GET  /health": "Health status"
        }
    }


@app.get("/health", tags=["Health"])
def health():
    """Returns health status and config."""
    return {
        "status": "healthy",
        "api_key_set": bool(os.getenv("GOOGLE_API_KEY")),
        "model": "gemini-2.5-flash",
        "agent": "tools_agent"
    }


@app.get("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat_get(query: str = Query(..., min_length=1, max_length=1000)):
    """
    Chat with the AI agent via GET request.
    Example: /chat?query=what is the weather in hyderabad
    """
    try:
        result = await run_agent(query)
        return result
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat_post(request: ChatRequest):
    """
    Chat with the AI agent via POST request.
    Body: { "query": "your question here" }
    """
    try:
        result = await run_agent(request.query)
        return result
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
