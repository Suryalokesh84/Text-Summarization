import pytesseract
from PIL import Image
from transformers import pipeline
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import shutil
import os

app = FastAPI()

# Allow frontend to access the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (Frontend)
app.mount("/frontend", StaticFiles(directory="../frontend"), name="frontend")

# Load the summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def extract_text_from_image(image_path):
    """Extracts text from an image using Pytesseract."""
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text.strip()

def summarize_text(text):
    """Summarizes text with dynamic length adjustments."""
    input_length = len(text.split())  # Word count
    max_len = min(1000, input_length // 2)  # Ensure summary is shorter
    min_len = min(30, max_len // 2)

    summary = summarizer(text, max_length=max_len, min_length=min_len, do_sample=False)
    return summary[0]["summary_text"]
    

@app.post("/summarize")
async def summarize_endpoint(text: str = Form(None), file: UploadFile = File(None)):
    """Handles both text input and image file upload for summarization."""
    if text:
        extracted_text = text
    elif file:
        file_path = f"temp_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        extracted_text = extract_text_from_image(file_path)
        os.remove(file_path)  # Clean up temp file
    else:
        return {"error": "Please provide text or an image."}

    if not extracted_text.strip():
        return {"error": "No readable text found in the image."}

    summary = summarize_text(extracted_text)
    return {"summary": summary}

@app.get("/")
async def serve_frontend():
    """Serves the frontend index.html file."""
    return FileResponse("../frontend/index.html")
