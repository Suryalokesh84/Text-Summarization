from fastapi import FastAPI, UploadFile, File
import pytesseract
from PIL import Image
from transformers import pipeline
import io

# Initialize FastAPI app
app = FastAPI()

# Load the summarization model (using a smaller model for speed)
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def extract_text_from_image(image_file):
    """Extracts text from an uploaded image using Pytesseract."""
    image = Image.open(image_file)
    text = pytesseract.image_to_string(image)
    return text.strip()

def summarize_text(text, max_length=100, min_length=30):
    """Summarizes text using a Transformer model."""
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]["summary_text"]

@app.post("/summarize-image")
async def summarize_from_image(file: UploadFile = File(...)):
    """API Endpoint: Upload an image, extract text, and return summarized text."""
    image_bytes = await file.read()
    image = io.BytesIO(image_bytes)

    extracted_text = extract_text_from_image(image)
    if not extracted_text:
        return {"error": "No text found in the image!"}

    summarized_text = summarize_text(extracted_text)
    return {
        "extracted_text": extracted_text,
        "summarized_text": summarized_text
    }

@app.post("/summarize-text")
async def summarize_from_text(text: str):
    """API Endpoint: Receive text and return summarized text."""
    if not text.strip():
        return {"error": "Empty text provided!"}

    summarized_text = summarize_text(text)
    return {
        "original_text": text,
        "summarized_text": summarized_text
    }

@app.get("/")
async def home():
    """Home Route"""
    return {"message": "Welcome to the Text Summarization API!"}
