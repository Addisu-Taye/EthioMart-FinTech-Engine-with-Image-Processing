import cv2
import pytesseract
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import re
from transformers import pipeline
from src.utils import normalize_amharic

class ImageProcessor:
    def __init__(self):
        # Initialize OCR (Tesseract) with Amharic language support
        pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # Update path as needed
        self.ocr_config = r'--oem 3 --psm 6 -l amh+eng'
        
        # Initialize product detection model (example using Hugging Face)
        self.product_detector = pipeline(
            "image-classification", 
            model="google/vit-base-patch16-224"
        )
    
    def preprocess_image(self, image_path):
        """Enhance image for better OCR results"""
        try:
            # Load image either from file or URL
            if image_path.startswith('http'):
                response = requests.get(image_path)
                img = Image.open(BytesIO(response.content))
            else:
                img = Image.open(image_path)
            
            # Convert to OpenCV format
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            
            # Preprocessing steps
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
            _, threshold = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            return threshold
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None
    
    def extract_text_from_image(self, image_path):
        """Extract Amharic and English text from images"""
        processed_img = self.preprocess_image(image_path)
        if processed_img is None:
            return ""
        
        # Perform OCR
        text = pytesseract.image_to_string(processed_img, config=self.ocr_config)
        return normalize_amharic(text)
    
    def detect_products_in_image(self, image_path):
        """Detect products in images using a pre-trained model"""
        try:
            if image_path.startswith('http'):
                response = requests.get(image_path)
                img = Image.open(BytesIO(response.content))
            else:
                img = Image.open(image_path)
            
            # Get predictions
            predictions = self.product_detector(img)
            
            # Filter relevant product predictions (confidence > 0.5)
            products = [
                pred for pred in predictions 
                if pred['score'] > 0.5 and 
                pred['label'] not in ['background', 'text', 'pattern']
            ]
            
            return [p['label'] for p in products[:3]]  # Return top 3 products
        except Exception as e:
            print(f"Error detecting products: {e}")
            return []
    
    def process_image_entities(self, image_path):
        """Extract entities from an image"""
        results = {
            'text': '',
            'products': [],
            'prices': [],
            'locations': []
        }
        
        # Extract text
        text = self.extract_text_from_image(image_path)
        results['text'] = text
        
        # Extract prices from text
        price_pattern = re.compile(r'(\d+[\s,]*\d*)\s*(ብር|ETB|Br|birr)', re.IGNORECASE)
        results['prices'] = [match.group(1) for match in price_pattern.finditer(text)]
        
        # Extract locations from text
        loc_patterns = [
            re.compile(r'(አዲስ[\s-]*አበባ|Addis Ababa|AA)', re.IGNORECASE),
            re.compile(r'(መቀሌ|Mekelle|Mekele)', re.IGNORECASE),
        ]
        results['locations'] = []
        for pattern in loc_patterns:
            results['locations'].extend(match.group(0) for match in pattern.finditer(text))
        
        # Detect products in image
        results['products'] = self.detect_products_in_image(image_path)
        
        return results

# Example usage
if __name__ == "__main__":
    processor = ImageProcessor()
    
    # Test with a sample image
    sample_image = "data/raw/media/sample_product.jpg"
    results = processor.process_image_entities(sample_image)
    
    print("Extracted Text:", results['text'])
    print("Detected Products:", results['products'])
    print("Found Prices:", results['prices'])
    print("Found Locations:", results['locations'])