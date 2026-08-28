import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from google import genai

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
                                                                #uvicorn app:app --reload
API_KEY = os.getenv("GEMINI_API_KEY")DATA_FILE = "knowledge.txt"

if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        f.write("Qubexa is a modern tech and software solutions enterprise.\nFounder and Owner: Rushikesh Gomsale.")

loader = TextLoader(DATA_FILE, encoding="utf-8")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=60)
docs = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
vector_db = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

client = genai.Client(api_key=API_KEY)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")

class ChatRequest(BaseModel):
    query: str

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    relevant_docs = retriever.invoke(request.query)
    context_text = "\n\n".join(doc.page_content for doc in relevant_docs)

    prompt = f"""You are a helpful AI assistant. Use the provided context to answer the user question accurately.
If the information is not present in the context, reply with 'Information not available'.

Context:
{context_text}

Question: {request.query}
Answer:"""

    try:
        response = client.models.generate_content(
        model="gemini-3.6-flash",            
        contents=prompt,
        )
        return {"answer": response.text}
    except Exception as e:
        return {"answer": f"Error: {str(e)}"}