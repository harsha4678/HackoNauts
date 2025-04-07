import google.generativeai as genai
import os
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Tesseract path for Windows
if os.name == 'nt':  # Windows
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Configure Gemini API
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

def extract_text_from_image(image_path):
    """Extract text from image using OCR"""
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        return f"Error extracting text from image: {str(e)}"

def extract_text_from_pdf(pdf_path):
    """Convert PDF to images and extract text"""
    try:
        images = convert_from_path(pdf_path)
        text = ""
        for image in images:
            text += pytesseract.image_to_string(image)
        return text
    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}"

def is_health_related(text):
    """Check if the text is related to health/medical content"""
    health_keywords = [
        'diagnosis', 'prescribed', 'medicine', 'tablet', 'capsule', 'dose',
        'mg', 'ml', 'prescription', 'patient', 'treatment', 'doctor', 'hospital',
        'clinic', 'symptoms', 'medical'
    ]
    text_lower = text.lower()
    matches = sum(1 for keyword in health_keywords if keyword in text_lower)
    return matches >= 2  # Require at least 2 health-related keywords

def analyze_prescription(file_path):
    """Analyze prescription using Gemini"""
    try:
        # Extract text based on file type
        file_ext = file_path.lower().split('.')[-1]
        if file_ext in ['jpg', 'jpeg', 'png']:
            prescription_text = extract_text_from_image(file_path)
        elif file_ext == 'pdf':
            prescription_text = extract_text_from_pdf(file_path)
        else:
            return {"error": "Unsupported file format"}

        # Check if content is health-related
        if not is_health_related(prescription_text):
            return {
                "success": False,
                "error": "Disclaimer: This document does not appear to be a medical prescription or health-related document. Please ensure you're uploading a valid medical prescription.",
                "original_text": prescription_text
            }

        # Prompt for analysis
        prompt = f"""
Please analyze this prescription and provide a detailed breakdown:
{prescription_text}

Please provide the following information:
1. Diagnosis/Condition
2. Prescribed Medications with dosage and duration
3. Special Instructions
4. Treatment Plan
5. Precautions and Side Effects
6. Follow-up Requirements

Format the response in a clear, structured manner.
"""

        response = genai.GenerativeModel('gemini-2.0-flash').generate_content(prompt)
        return {
            "success": True,
            "analysis": response.text,
            "original_text": prescription_text
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
