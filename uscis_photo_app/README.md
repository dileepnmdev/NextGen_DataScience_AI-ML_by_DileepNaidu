# USCIS Passport Photo App

This folder contains a simple proof of concept web application that allows users to upload an image and automatically resize it to meet the USCIS passport photo requirement of 2x2 inches (600x600 pixels).

## Architecture Overview

```
Browser -> Flask App -> Pillow Image Processing -> Download result
```

1. **Frontend (HTML form)** – A minimal page with a file input allows the user to select an image from their device.
2. **Backend (Flask)** – Receives the file upload, uses Pillow to resize and crop the image to exactly 600x600 pixels with a white background, then returns the processed file for download.
3. **Ads (placeholder)** – The page template includes a placeholder section where advertising code (such as Google AdSense) could be inserted.

## USCIS Photo Requirements Implemented

- Final image size is 2x2 inches (51x51mm) which translates to 600x600 pixels at 300 DPI.
- Photo is in color (JPG) with a plain white background.
- The output retains the original aspect ratio by filling empty space with white if needed.

Additional USCIS guidelines (face size, no glasses, neutral expression, etc.) are documented but not programmatically enforced in this demo.

## Running the App

```
pip install -r ../../requirements.txt  # from repository root
python app.py
```

Navigate to `http://localhost:5000` and upload a photo to test.
