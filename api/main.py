from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import requests
import json

from starlette.middleware.cors import CORSMiddleware

# Create a FastAPI app instance
app = FastAPI()

#Enabling CORS
origins =[
    "http://localhost",
    "http://localhost:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# TensorFlow Serving endpoint
endpoint = "http://127.0.0.1:8502/v1/models/potato_model:predict"

# Class names of the model output
Class_names = ["Early Blight", "Late Blight", "Healthy"]

# Image input dimensions expected by the model
IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 3  # RGB

# Test endpoint to check if API is working
@app.get("/ping")
async def ping():
    return {"message": "Hello World! FastAPI is running."}

# Function to convert uploaded image file into a numpy array
def read_file_as_image(data: bytes) -> np.ndarray:
    try:
        image = Image.open(BytesIO(data))

        # Convert image to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize the image
        image = image.resize((IMG_WIDTH, IMG_HEIGHT))

        # Convert to numpy array
        image_array = np.array(image)

        # Normalize pixel values to 0-1
        image_array = image_array / 255.0

        # Convert to float32
        image_array = image_array.astype(np.float32)

        return image_array
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {e}")

# Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read uploaded image file
        image_bytes = await file.read()

        # Preprocess the image
        image = read_file_as_image(image_bytes)

        # Add batch dimension
        img_batch = np.expand_dims(image, axis=0)

        # Create JSON data for TF Serving
        json_data = {"instances": img_batch.tolist()}

        # Send request to TensorFlow Serving
        response = requests.post(endpoint, json=json_data, timeout=60)

        # Raise error if request failed
        response.raise_for_status()

        # Parse predictions
        predictions = response.json().get("predictions")
        if not predictions or not isinstance(predictions, list) or not predictions[0]:
            raise HTTPException(status_code=500, detail="Invalid prediction response from TensorFlow Serving.")

        # Extract prediction and confidence
        prediction = np.array(predictions[0])
        predicted_class = Class_names[np.argmax(prediction)]
        confidence = np.max(prediction)

        return {
            "class_name": predicted_class,
            "confidence": float(confidence)
        }

    except requests.exceptions.ConnectionError as e:
        raise HTTPException(status_code=503, detail=f"Could not connect to TensorFlow Serving: {e}")
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Request to TensorFlow Serving timed out.")
    except requests.exceptions.RequestException as e:
        error_detail = f"Error from TensorFlow Serving: {e}"
        if response and response.text:
            try:
                tf_serving_error = json.loads(response.text)
                error_detail = tf_serving_error.get('error', response.text)
            except json.JSONDecodeError:
                error_detail = response.text
        raise HTTPException(status_code=response.status_code if response else 500, detail=error_detail)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

# Start the FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
