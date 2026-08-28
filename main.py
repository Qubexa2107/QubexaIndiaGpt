import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai

load_dotenv()

app = FastAPI(title="QubexaIndiaGpt API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.getenv("GEMINI_API_KEY")

SYSTEM_INSTRUCTION = """
You are 'QubexaIndiaGpt' - the official AI assistant of Qubexa.
Company Profile:
- Developer & Owner: Qubexa (Founder & CEO: Rushikesh Gomsale).
- Core Domains: Next-generation tech solutions, cybersecurity, AI architectures, custom software development, and IoT research.
"""

class ChatRequest(BaseModel):
    query: str = ""
    message: str = ""

class ChatResponse(BaseModel):
    reply: str

@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_qubexa(request: ChatRequest):
    user_input = request.query or request.message
    if not user_input:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    if not API_KEY:
        print("Backend Execution Error: GEMINI_API_KEY is missing on Render Environment Variables.")
        raise HTTPException(status_code=500, detail="Server API key is not configured.")

    try:
        client = genai.Client(api_key=API_KEY)
        
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=f"System Instruction: {SYSTEM_INSTRUCTION}\n\nUser Question: {user_input}"
        )
        return ChatResponse(reply=response.text)
    except Exception as e:
        print(f"Backend Execution Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health_check():
    return {
        "status": "Online",
        "service": "QubexaIndiaGpt Backend",
        "version": "1.0"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)