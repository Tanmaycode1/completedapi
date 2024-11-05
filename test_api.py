import requests
import time
import json
from pathlib import Path

def test_pdf_processing(pdf_path: str, api_base_url: str = "http://localhost:8000"):
    """
    Test the PDF processing API with a scanned PDF file
    
    Args:
        pdf_path: Path to the PDF file
        api_base_url: Base URL of the FastAPI server
    """
    # Check if file exists
    if not Path(pdf_path).exists():
        print(f"Error: File not found at {pdf_path}")
        return
        
    try:
        # Prepare the file for upload
        with open(pdf_path, 'rb') as pdf_file:
            files = {'file': (Path(pdf_path).name, pdf_file, 'application/pdf')}
            
            # Additional parameters
            data = {
                'prompt_type': 'policy_json_conversion',
                'ocr_language': 'eng',  # Use 'eng+fra' for English+French, etc.
                'force_ocr': 'true'     # Force OCR even if text seems extractable
            }
            
            print(f"\n1. Uploading PDF: {Path(pdf_path).name}")
            # Submit the PDF for processing
            response = requests.post(
                f"{api_base_url}/api/process-pdf",
                files=files,
                data=data
            )
            
            if response.status_code != 200:
                print(f"Error submitting PDF: {response.text}")
                return
                
            task_data = response.json()
            task_id = task_data['task_id']
            print(f"Task ID: {task_id}")
            
            # Poll for results
            print("\n2. Waiting for processing to complete...")
            while True:
                status_response = requests.get(f"{api_base_url}/api/pdf-status/{task_id}")
                
                if status_response.status_code != 200:
                    print(f"Error checking status: {status_response.text}")
                    break
                    
                status_data = status_response.json()
                status = status_data['status']
                
                # Print progress
                if status == 'processing':
                    progress = status_data.get('progress', 0)
                    pages_processed = status_data.get('pages_processed', 0)
                    total_pages = status_data.get('total_pages', '?')
                    print(f"\rProgress: {progress:.1f}% (Page {pages_processed}/{total_pages})", end='')
                
                # Check if processing is complete
                if status == 'completed':
                    print("\n\n3. Processing completed!")
                    print("\nResults:")
                    print("-" * 50)
                    print(f"OCR Used: {status_data.get('ocr_used', False)}")
                    print(f"Total Pages: {status_data.get('total_pages', 0)}")
                    print(f"Processing Time: {status_data.get('processing_time', 0):.2f} seconds")
                    print(f"Word Count: {status_data.get('word_count', 0)}")
                    print("\nExtracted Text Sample (first 500 chars):")
                    print("-" * 50)
                    print(status_data.get('text_content', '')[:500])
                    print("-" * 50)
                    
                    # Save full text to file
                    output_file = Path(pdf_path).stem + "_extracted_text.txt"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(status_data.get('text_content', ''))
                    print(f"\nFull text saved to: {output_file}")
                    break
                    
                elif status == 'failed':
                    print(f"\nProcessing failed: {status_data.get('error', 'Unknown error')}")
                    break
                    
                time.sleep(1)  # Wait before checking again
                
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    # Example usage
    pdf_file_path = "normal.pdf"  # Replace with your PDF path
    test_pdf_processing(pdf_file_path)