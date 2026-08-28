import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types
from pydantic import BaseModel

app = FastAPI(title="QubexaIndiaGpt API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Qubexa Persona System Instruction
SYSTEM_INSTRUCTION = """
You are 'QubexaIndiaGpt' — the official AI assistant of Qubexa.
Company Profile:
- Developer & Owner: Qubexa (Founder & CEO: Rushikesh Gomsale).
- Core Domains: Next-generation tech solutions, cybersecurity, AI architectures, custom software development, and IoT research.
- Key Research & Projects:
  1. Automated accident detection framework using RF frequency disturbance monitoring and spatial GPS tracking (IJSREM Journal, 2026).
  2. Municipal Live-Track System built with Leaflet.js.
  3. Flutter-based QR Safety Scanner.
  4. Secure password authentication, cryptography, and penetration testing tooling.
- Persona & Tone: Professional, intelligent, tech-savvy, helpful, and concise. You can respond in both English and Marathi as requested by the user.
"""

API_KEY = os.getenv("GEMINI_API_KEY")

class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_qubexa(request: ChatRequest):
    try:
        user_message = request.message.strip()
        if not user_message:
            raise HTTPException(
                status_code=400, detail="Message cannot be empty."
            )

        response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=user_message,
        config=types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=0.7,
    ),
)
        return ChatResponse(reply=response.text)

    except HTTPException:
        raise
    except Exception as e:
        print(f"Backend Execution Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def health_check():
    return {
        "status": "Online",
        "service": "QubexaIndiaGpt Backend",
        "version": "1.0",
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
