from flask import Flask, render_template, request, send_file
import os
import cv2
import numpy as np
from compression.compression import compress_image

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
COMPRESSED_FOLDER = "compressed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(COMPRESSED_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compress', methods=['POST'])
def compress_image_route():
    if 'file' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    # Save the uploaded file to the 'uploads' folder
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Read the image using OpenCV
    input_image = cv2.imread(filepath)

    # Get the quality parameter from the form, default to 80 if not provided
    quality = int(request.form.get('quality', 80))

    # Compress the image using the compress_image function
    compressed_image = compress_image(input_image, quality)

    # Save the compressed image to the 'compressed' folder
    compressed_path = os.path.join(COMPRESSED_FOLDER, "compressed_" + file.filename)
    cv2.imwrite(compressed_path, compressed_image)

    # Send the compressed image as a response for download
    return send_file(compressed_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
