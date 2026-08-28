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

# API Key लोड करा आणि Google GenAI Client तयार करा
API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)  # <-- ही लाइन जोडणे आवश्यक आहे

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
        raise HTTPException(status_code=400, detail="Query or message cannot be empty.")

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_input,
            config=dict(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.7
            )
        )
        return ChatResponse(reply=response.text)
    except Exception as e:
        print(f"Backend Execution Error: {e}")
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