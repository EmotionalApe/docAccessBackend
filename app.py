from flask import Flask, request, jsonify
import base64
from PIL import Image
from io import BytesIO
import cv2
import numpy as np
import io

import easyocr
import cv2
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image


app = Flask(__name__)

@app.route("/")
def home():
    return "Backend for Accessibility app"

@app.route('/deuteranopia', methods=['POST'])
def deuteranopia():
    try:
        # Receive image data from React Native app
        data = request.get_json()
        image_base64 = data['image'].split(",")
        image_base64 = image_base64[1]

        #image_bytes = BytesIO(base64.b64decode(image_base64))
        image = Image.open(io.BytesIO(base64.decodebytes(bytes(image_base64, "utf-8"))))


        #image = Image.open(image_bytes)

        # Convert PIL image to OpenCV format
        cv2_image = np.array(image)
        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_RGB2BGR)

        # Perform custom image processing
        processed_image = deuteranopia_internal(cv2_image)

        # Convert the processed image back to PIL format
        processed_pil_image = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))

        # Save the processed image to a BytesIO object
        processed_image_bytes_io = BytesIO()
        processed_pil_image.save(processed_image_bytes_io, format='JPEG')

        # Encode the processed image to base64
        processed_image_base64 = base64.b64encode(processed_image_bytes_io.getvalue()).decode('utf-8')

        return jsonify({'processedImage': processed_image_base64})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/protanopia', methods=['POST'])
def protanopia():
    try:
        # Receive image data from React Native app
        data = request.get_json()
        image_base64 = data['image'].split(",")
        image_base64 = image_base64[1]

        #image_bytes = BytesIO(base64.b64decode(image_base64))
        image = Image.open(io.BytesIO(base64.decodebytes(bytes(image_base64, "utf-8"))))


        #image = Image.open(image_bytes)

        # Convert PIL image to OpenCV format
        cv2_image = np.array(image)
        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_RGB2BGR)

        # Perform custom image processing
        processed_image = protanopia_internal(cv2_image)

        # Convert the processed image back to PIL format
        processed_pil_image = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))

        # Save the processed image to a BytesIO object
        processed_image_bytes_io = BytesIO()
        processed_pil_image.save(processed_image_bytes_io, format='JPEG')

        # Encode the processed image to base64
        processed_image_base64 = base64.b64encode(processed_image_bytes_io.getvalue()).decode('utf-8')

        return jsonify({'processedImage': processed_image_base64})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/tritanopia', methods=['POST'])
def tritanopia():
    try:
        # Receive image data from React Native app
        data = request.get_json()
        image_base64 = data['image'].split(",")
        image_base64 = image_base64[1]

        #image_bytes = BytesIO(base64.b64decode(image_base64))
        image = Image.open(io.BytesIO(base64.decodebytes(bytes(image_base64, "utf-8"))))


        #image = Image.open(image_bytes)

        # Convert PIL image to OpenCV format
        cv2_image = np.array(image)
        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_RGB2BGR)

        # Perform custom image processing
        processed_image = tritanopia_internal(cv2_image)

        # Convert the processed image back to PIL format
        processed_pil_image = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))

        # Save the processed image to a BytesIO object
        processed_image_bytes_io = BytesIO()
        processed_pil_image.save(processed_image_bytes_io, format='JPEG')

        # Encode the processed image to base64
        processed_image_base64 = base64.b64encode(processed_image_bytes_io.getvalue()).decode('utf-8')

        return jsonify({'processedImage': processed_image_base64})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/monochromacy', methods=['POST'])
def monochromacy():
    try:
        # Receive image data from React Native app
        data = request.get_json()
        image_base64 = data['image'].split(",")
        image_base64 = image_base64[1]

        #image_bytes = BytesIO(base64.b64decode(image_base64))
        image = Image.open(io.BytesIO(base64.decodebytes(bytes(image_base64, "utf-8"))))


        #image = Image.open(image_bytes)

        # Convert PIL image to OpenCV format
        cv2_image = np.array(image)
        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_RGB2BGR)

        # Perform custom image processing
        processed_image = monochromacy_internal(cv2_image)

        # Convert the processed image back to PIL format
        processed_pil_image = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))

        # Save the processed image to a BytesIO object
        processed_image_bytes_io = BytesIO()
        processed_pil_image.save(processed_image_bytes_io, format='JPEG')

        # Encode the processed image to base64
        processed_image_base64 = base64.b64encode(processed_image_bytes_io.getvalue()).decode('utf-8')

        return jsonify({'processedImage': processed_image_base64})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

      

@app.route('/blind', methods=['POST'])
def blind():
    try:
        # Receive image data from React Native app
        data = request.get_json()
        image_base64 = data['image'].split(",")
        image_base64 = image_base64[1]

        #image_bytes = BytesIO(base64.b64decode(image_base64))
        image = Image.open(io.BytesIO(base64.decodebytes(bytes(image_base64, "utf-8"))))

        #image = Image.open(image_bytes)

        # Convert PIL image to OpenCV format
        cv2_image = np.array(image)
        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_RGB2BGR)


        # Perform OCR on the image
        ocr_result = perform_ocr_internal(cv2_image)
        print("ocr_result")

        summarisedText = summarise(ocr_result)

        return jsonify({'summarisedText': summarisedText})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def deuteranopia_internal(image):
    # Your existing image processing logic
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 100, 100])
    upper_red = np.array([30, 255, 255])

    lower_green = np.array([40, 40, 40])
    upper_green = np.array([90, 255, 255])

    red_mask = cv2.inRange(hsv_image, lower_red, upper_red)
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

    hsv_image[red_mask > 0, 0] = 120
    hsv_image[green_mask > 0, 0] = 30

    transformed_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    return transformed_image

def protanopia_internal(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([30, 255, 255])

    
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([90, 255, 255])

    red_mask = cv2.inRange(hsv_image, lower_red, upper_red)
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

  
    hsv_image[red_mask > 0, 0] = 30  

    
    hsv_image[green_mask > 0, 0] = 120  

    # Convert the HSV image back to BGR color space
    transformed_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    return transformed_image

def tritanopia_internal(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([150, 255, 255])

    blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

    hsv_image[blue_mask > 0, 0] = 20 
    hsv_image[blue_mask > 0, 1] = 255  

    lower_green = np.array([40, 30, 30])
    upper_green = np.array([90, 255, 255])

    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

    hsv_image[green_mask > 0, 0] = 330 

    transformed_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    return transformed_image

def monochromacy_internal(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clip_limit=2.0
    tile_size=(8, 8)

    # Apply adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    enhanced_image = clahe.apply(grayscale_image)

    # Convert back to BGR for display and saving
    enhanced_image_bgr = cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2BGR)

    # Combine the original and enhanced images
    transformed_image = cv2.addWeighted(image, 0.5, enhanced_image_bgr, 0.5, 0)

    return transformed_image

def perform_ocr_internal(image):
    # Perform OCR using easyocr
    reader = easyocr.Reader({'en'}, gpu=True)
    result = reader.readtext(image)
    # Extract text from the OCR result
    ocrText = ""
    for i in range(len(result)):
        ocrText = ocrText + result[i][1]
    return ocrText

def summarise(ocrText):
  import requests
  print("summarisation started")
  API_URL = "https://api-inference.huggingface.co/models/slauw87/bart_summarisation"
  headers = {"Authorization": "Bearer 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"}

  response = requests.post(API_URL, headers=headers, json={"inputs":ocrText})
  response = response.json()
  output=response[0]['summary_text']
  print("summarisation done")
  return output


# if __name__ == "__main__":
#     app.run()