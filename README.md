# 🌽 AgroMaizeVision API

A FastAPI backend that powers **AgroMaizeVision**, an AI-powered maize disease detection application.

The API accepts images of maize leaves, processes them using a TensorFlow deep learning model, and returns the predicted disease class along with a confidence score.

---

## Features

- RESTful API built with FastAPI
- Image upload endpoint
- TensorFlow/Keras model inference
- Automatic model download from Hugging Face Hub
- Cross-Origin Resource Sharing (CORS) support
- JSON prediction responses
- Health check endpoint

---

## Machine Learning Pipeline

```
Client
   │
   ▼
Upload Image
   │
   ▼
FastAPI
   │
   ▼
TensorFlow Model
   │
   ▼
Prediction
   │
   ▼
JSON Response
```

---

## API Endpoints

### Health Check

```
GET /ping
```

Response

```json
"Hello, I am alive"
```

---

### Predict Disease

```
POST /predict
```

Upload

- multipart/form-data
- image file

Example Response

```json
{
  "class": "Common Rust",
  "confidence": 0.997
}
```

---

## Disease Classes

- Common Rust
- Gray Leaf Spot
- Healthy
- Northern Leaf Blight

---

## Technologies

- Python
- FastAPI
- TensorFlow
- Keras
- NumPy
- Pillow
- Hugging Face Hub
- Uvicorn

---

## Installation

Clone the repository

```bash
git clone https://github.com/williams-jpy/agromaizevision-api.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the server

```bash
python main.py
```

---

## Project Team

Project Lead

- Williams Nwachi

---

## License

MIT License
