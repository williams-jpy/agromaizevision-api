from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import tensorflow as tf
import io
import os
from huggingface_hub import hf_hub_download

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Download and load model from Hugging Face
model_path = hf_hub_download(
    repo_id="williams-jpy/agromaizevision",
    filename="3.keras"
)
model = tf.keras.models.load_model(model_path)

# Maize disease classes
class_names = [
    "Common Rust",
    "Gray Leaf Spot",
    "Healthy",
    "Northern Leaf Blight"
]

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(io.BytesIO(data)))
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = read_file_as_image(contents)
    img_batch = np.expand_dims(image, 0)

    predictions = model.predict(img_batch)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=7860)