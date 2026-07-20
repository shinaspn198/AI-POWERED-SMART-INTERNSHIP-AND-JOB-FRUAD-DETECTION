import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

img = Image.open(r"C:\Users\91949\OneDrive\Downloads\Desktop\my works\internship_fraud_detection\dataset\fake-job-offer-example.jpg")

text = pytesseract.image_to_string(img)
print("=== RAW OCR TEXT ===")
print(text)
print("=== WORD COUNT:", len(text.split()), "===")
print("=== KEY PHRASES ===")
for phrase in ["registration", "charges", "155", "72 hours", "factory"]:
    print(f"  {phrase}:", phrase.lower() in text.lower())
