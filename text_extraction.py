import pytesseract
from PIL import Image

def extract_text(image_path):
    """Extracts text from a printed document image using Pytesseract."""
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text

# Example usage
if __name__ == "__main__":
    image_path = "image.jpg"  # Replace with your image file
    extracted_text = extract_text(image_path)
    print("Extracted Text:\n", extracted_text)    
