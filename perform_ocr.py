import os
import csv
import cv2
import pytesseract
from tqdm import tqdm

def perform_ocr(processed_images_folder, ocr_output_csv):
    """
    Performs OCR on all images in the specified folder and writes the extracted text to a CSV file.
    """
    # Initialize Tesseract OCR
    # If Tesseract is not in your PATH, specify the path to tesseract executable
    # Example for Windows:
    pytesseract.pytesseract.tesseract_cmd = '/usr/local/opt/tesseract/bin/tesseract'
    
    with open(ocr_output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['image_filename', 'extracted_text'])
        
        # List all image files
        image_files = [f for f in os.listdir(processed_images_folder) if f.endswith('.jpg') or f.endswith('.png')]
        
        for img_filename in tqdm(image_files, desc="Performing OCR"):
            img_path = os.path.join(processed_images_folder, img_filename)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to read image {img_path}. Skipping.")
                continue
            
            # Preprocessing for better OCR accuracy
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 3)
            gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 31, 2)
            
            # Perform OCR using Tesseract
            extracted_text = pytesseract.image_to_string(gray, lang='eng', config='--psm 7')
            
            # Clean up the text
            extracted_text = extracted_text.strip()
            
            csv_writer.writerow([img_filename, extracted_text])

def main():
    # Paths to processed images and OCR output CSVs
    train_processed_images = 'input/gnhk_dataset/train_processed/images'
    test_processed_images = 'input/gnhk_dataset/test_processed/images'
    
    train_ocr_csv = 'input/gnhk_dataset/train_ocr.csv'
    test_ocr_csv = 'input/gnhk_dataset/test_ocr.csv'
    
    # Perform OCR on training images
    perform_ocr(train_processed_images, train_ocr_csv)
    
    # Perform OCR on testing images
    perform_ocr(test_processed_images, test_ocr_csv)

if __name__ == '__main__':
    main()
