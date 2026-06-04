from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import tensorflow as tf
import io

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your trained model
model = tf.keras.models.load_model(r"C:\Users\hilary\Documents\Model\3.keras")

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
    uvicorn.run(app, host='localhost', port=8000)