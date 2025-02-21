# OMR (Optical Mark Recognition) System

## Overview
This project is an Optical Mark Recognition (OMR) system that processes scanned exam sheets to extract student IDs and answers from multiple-choice and true/false questions. The system utilizes OpenCV for image processing and `pdf2image` for handling PDF inputs.

## Features
- Reads and processes scanned exam sheets
- Extracts student IDs from marked bubbles
- Detects and classifies answers for multiple-choice and true/false questions
- Saves processed images with detected markings for verification
- Logs detailed debugging information

## Dependencies
Ensure you have the following dependencies installed:
```sh
pip install opencv-python numpy pdf2image
```

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo/omr-system.git
   cd omr-system
   ```
2. Install required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
### Running the OMR System
Run the script with a test image:
```sh
python omr.py
```

### Processing an Exam Sheet
The system processes an exam sheet by:
1. Loading the image.
2. Preprocessing (grayscale conversion, thresholding, noise reduction).
3. Detecting and extracting marked bubbles.
4. Identifying student ID and answers.
5. Outputting results in the console and saving the processed image.

### Expected Output
A processed image with detected bubbles and results printed in the terminal, including:
```sh
SINAV SONUÇLARI
ÖĞRENCİ BİLGİLERİ:
Form ID: QL50XB4C
Öğrenci No: 123456

CEVAPLAR:
Soru 01: A
Soru 02: C
...
JSON FORMATI:
{
    "student_id": "123456",
    "form_id": "QL50XB4C",
    "answers": {
        "multiple_choice": {
            "01": "A",
            "02": "C"
        },
        "true_false": {
            "21": "T",
            "22": "F"
        }
    }
}
```

## File Structure
```
|-- omr.py               # Main script for OMR processing
|-- test.png             # Sample exam sheet image
|-- processed_exam.png   # Output image with detected markings
|-- requirements.txt     # List of dependencies
```

## Troubleshooting
- **Issue:** Image not loading properly  
  **Solution:** Ensure the file path is correct and the image format is supported.
- **Issue:** No bubbles detected  
  **Solution:** Adjust preprocessing parameters like thresholding and contour detection.
- **Issue:** Student ID or answers incorrect  
  **Solution:** Verify the image quality and contrast.

## Future Improvements
- Support for different exam formats
- Enhanced error correction for misread markings
- Web interface for uploading and processing sheets

## License
This project is licensed under the MIT License.

## Author
Developed by [Your Name](https://github.com/your-profile).

