import pytesseract
from PIL import Image
from transformers import pipeline

# Load the summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def extract_text(image_path):
    """Extracts text from a printed document image using Pytesseract."""
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text.strip()

def summarize_text(text, max_length=100, min_length=30):
    """Summarizes extracted text using a Transformer model (BART)."""
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]["summary_text"]

# Example usage
if __name__ == "__main__":
    image_path = "image.jpg"  # Replace with your image
    extracted_text = extract_text(image_path)
    
    print("\n🔹 Extracted Text:\n", extracted_text)
    
    if extracted_text:
        summarized_text = summarize_text(extracted_text)
        print("\n🔹 Summarized Text:\n", summarized_text)
