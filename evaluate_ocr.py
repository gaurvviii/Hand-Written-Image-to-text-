import pandas as pd
from jiwer import wer

def evaluate_ocr(ground_truth_csv, ocr_output_csv, report_path=None):
    """
    Evaluates OCR results by comparing extracted text with ground truth.
    Calculates Word Error Rate (WER).
    """
    # Load ground truth and OCR output
    ground_df = pd.read_csv(ground_truth_csv)
    ocr_df = pd.read_csv(ocr_output_csv)
    
    # Merge on image_filename
    merged_df = pd.merge(ground_df, ocr_df, on='image_filename', how='inner')
    
    # Calculate WER for each entry
    merged_df['wer'] = merged_df.apply(lambda row: wer(row['text'], row['extracted_text']), axis=1)
    
    # Calculate average WER
    average_wer = merged_df['wer'].mean()
    
    print(f"Average WER: {average_wer:.4f}")
    
    # Optionally, save the detailed report
    if report_path:
        merged_df.to_csv(report_path, index=False)
        print(f"Detailed report saved to {report_path}")

def main():
    # Paths to ground truth and OCR output CSVs
    train_ground_truth = 'input/gnhk_dataset/train_processed.csv'
    train_ocr_output = 'input/gnhk_dataset/train_ocr.csv'
    
    test_ground_truth = 'input/gnhk_dataset/test_processed.csv'
    test_ocr_output = 'input/gnhk_dataset/test_ocr.csv'
    
    # Evaluate training data
    print("Evaluating Training Data:")
    evaluate_ocr(train_ground_truth, train_ocr_output, report_path='input/gnhk_dataset/train_evaluation.csv')
    
    # Evaluate testing data
    print("Evaluating Testing Data:")
    evaluate_ocr(test_ground_truth, test_ocr_output, report_path='input/gnhk_dataset/test_evaluation.csv')

if __name__ == '__main__':
    main()
