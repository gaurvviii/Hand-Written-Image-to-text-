import os
import json
import csv
import cv2
import numpy as np
from tqdm import tqdm

def create_directories():
    dirs = [
        'input/gnhk_dataset/train_processed/images',
        'input/gnhk_dataset/test_processed/images',
        'input/gnhk_dataset/train_data/train',
        'input/gnhk_dataset/test_data/test'
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

def polygon_to_bbox(polygon):
    """
    Converts a polygon (with keys x0, y0, x1, y1, x2, y2, x3, y3) to a bounding box (x, y, w, h).
    """
    try:
        points = np.array([(polygon[f'x{i}'], polygon[f'y{i}']) for i in range(4)], dtype=np.int32)
        x, y, w, h = cv2.boundingRect(points)
        return x, y, w, h
    except KeyError as e:
        print(f"Missing key in polygon: {e}")
        return None

def process_dataset(input_folder, output_folder, csv_path):
    """
    Processes the dataset by cropping images based on bounding boxes and writing the mappings to a CSV file.
    """
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['image_filename', 'text'])
        
        # Iterate over all JSON files in the input folder
        for filename in tqdm(os.listdir(input_folder), desc=f"Processing {os.path.basename(input_folder)}"):
            if filename.endswith('.json'):
                json_path = os.path.join(input_folder, filename)
                img_filename = filename.replace('.json', '.jpg')
                img_path = os.path.join(input_folder, img_filename)
                
                if not os.path.exists(img_path):
                    print(f"Image file {img_path} does not exist. Skipping.")
                    continue
                
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Failed to read image {img_path}. Skipping.")
                    continue
                
                for idx, item in enumerate(data):
                    text = item.get('text', '')
                    if text.startswith('%') and text.endswith('%'):
                        text = 'SPECIAL_CHARACTER'
                    
                    polygon = item.get('polygon', {})
                    if not all(f'x{i}' in polygon and f'y{i}' in polygon for i in range(4)):
                        print(f"Invalid polygon in {json_path}, item {idx}. Skipping.")
                        continue
                    bbox = polygon_to_bbox(polygon)
                    if bbox is None:
                        continue
                    x, y, w, h = bbox
                    
                    # Ensure bounding box is within image boundaries
                    x = max(x, 0)
                    y = max(y, 0)
                    w = min(w, img.shape[1] - x)
                    h = min(h, img.shape[0] - y)
                    
                    cropped_img = img[y:y+h, x:x+w]
                    
                    output_filename = f"{os.path.splitext(filename)[0]}_{idx}.jpg"
                    output_path = os.path.join(output_folder, output_filename)
                    cv2.imwrite(output_path, cropped_img)
                    
                    csv_writer.writerow([output_filename, text])

def main():
    # Step 1: Create necessary directories
    create_directories()
    
    # Step 2: Process training data
    process_dataset(
        'input/gnhk_dataset/train_data/train',
        'input/gnhk_dataset/train_processed/images',
        'input/gnhk_dataset/train_processed.csv'
    )
    
    # Step 3: Process testing data
    process_dataset(
        'input/gnhk_dataset/test_data/test',
        'input/gnhk_dataset/test_processed/images',
        'input/gnhk_dataset/test_processed.csv'
    )

if __name__ == '__main__':
    main()
