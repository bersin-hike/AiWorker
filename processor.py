
import fitz  # PyMuPDF
import os
import re
import io
import logging
from PIL import Image
import pytesseract

# --- SETUP LOGGING ---
logger = logging.getLogger("Processor")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

# --- CONFIGURATION ---
# If you are testing locally on Windows, you might need to uncomment and set this:
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def clean_text(text):
    """
    Basic text cleaning to normalize whitespace.
    """
    if not text: return ""
    # Replace multiple newlines/tabs/spaces with a single space
    # but preserve paragraph breaks (\n\n)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text) 
    return text.strip()

def get_text_via_tesseract(img_bytes):
    """
    Uses Tesseract OCR to extract text from an image.
    Replaces Gemini Vision for 100% free processing.
    """
    try:
        # Load image from bytes
        image = Image.open(io.BytesIO(img_bytes))
        
        # --oem 3: Default LSTM engine
        # --psm 3: Fully automatic page segmentation (good for docs with columns/tables)
        custom_config = r'--oem 3 --psm 3' 
        
        text = pytesseract.image_to_string(image, config=custom_config)
        return text.strip()
    except Exception as e:
        logger.error(f"❌ Tesseract OCR Error: {e}")
        return ""

def process_file(file_input, filename):
    """
    Hybrid Processor: Handles Text PDFs (Fast), Scanned PDFs (OCR), and Images (OCR).
    """
    text_content = ""
    filename_lower = filename.lower()

    try:
        # --- CASE 1: PDF Handling ---
        if filename_lower.endswith(".pdf"):
            
            # Open PDF (Handle both path string and file object)
            if isinstance(file_input, str):
                doc = fitz.open(file_input)
            else:
                file_input.seek(0)
                doc = fitz.open(stream=file_input.read(), filetype="pdf")

            page_count = 0
            for page in doc:
                page_count += 1
                
                # 1. Try standard text extraction first (Fast & Free)
                raw_text = page.get_text()
                
                # 2. Check Quality (Is it Scanned?)
                # Logic: If text is empty or very short, it's likely a scan/image.
                if len(raw_text.strip()) < 50: 
                    logger.info(f"🔍 Page {page_count} looks scanned. Running Tesseract OCR...")
                    
                    # 🚀 TRIGGER TESSERACT OCR
                    # matrix=fitz.Matrix(2, 2) doubles the resolution (Zoom x2)
                    # This is CRITICAL for Tesseract to read small text accurately.
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    img_data = pix.tobytes("png")
                    
                    ocr_text = get_text_via_tesseract(img_data)
                    text_content += f"\n{clean_text(ocr_text)}\n"
                
                else:
                    # Use standard extracted text
                    text_content += f"\n{clean_text(raw_text)}\n"
            
            doc.close()

        # --- CASE 2: Image Files (JPG/PNG) ---
        elif filename_lower.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
            
            if isinstance(file_input, str):
                with open(file_input, "rb") as f:
                    img_bytes = f.read()
            else:
                file_input.seek(0)
                img_bytes = file_input.read()
                
            logger.info(f"🔍 Running OCR on image: {filename}")
            ocr_text = get_text_via_tesseract(img_bytes)
            text_content = clean_text(ocr_text)
        
        # --- CASE 3: Text Files ---
        elif filename_lower.endswith('.txt'):
             if isinstance(file_input, str):
                with open(file_input, 'r', encoding='utf-8') as f:
                    text_content = f.read()
             else:
                file_input.seek(0)
                text_content = file_input.read().decode('utf-8')

    except Exception as e:
        logger.error(f"❌ Processing Error in {filename}: {e}")

    return text_content

def chunk_text(text, chunk_size=1000, overlap=100):
    """
    Your FIXED CHUNKING LOGIC (Preserved Exactly).
    """
    if not text:
        return []

    # Standardize newlines
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Strategy 1: Split by Paragraphs
    splits = text.split('\n\n')
    
    # Strategy 2: If paragraphs didn't work (weird PDF format), try lines
    if len(splits) < 2:
        splits = text.split('\n')

    # Strategy 3: If lines didn't work, try sentences
    if len(splits) < 2:
        splits = text.split('. ')

    final_chunks = []
    current_chunk = ""

    for split in splits:
        split = split.strip()
        if not split:
            continue

        # If adding this piece makes the chunk too big, save current and start new
        if len(current_chunk) + len(split) > chunk_size:
            if current_chunk:
                final_chunks.append(current_chunk.strip())
            # Start new chunk with overlap
            current_chunk = current_chunk[-overlap:] + " " + split
        else:
            # Add to current chunk
            current_chunk += "\n" + split

    # Add the last chunk
    if current_chunk:
        final_chunks.append(current_chunk.strip())

    return final_chunks