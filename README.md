# 🌽 AgroMaizeVision API

A FastAPI backend that powers **AgroMaizeVision**, an AI-driven application for detecting maize leaf diseases from uploaded images.

---

## Overview

The AgroMaizeVision API receives images from the frontend, processes them using a machine learning model, and returns disease predictions through RESTful endpoints.

This project was developed as part of a collaborative team effort, with **Williams Nwachi serving as Project Lead**, coordinating development and contributing to the implementation of the backend architecture.

---

## Features

- RESTful API built with FastAPI
- Image upload handling
- Machine learning inference
- JSON responses
- Frontend integration
- High-performance asynchronous framework

---

## Tech Stack

- Python
- FastAPI
- Uvicorn
- Machine Learning

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

Start the server

```bash
uvicorn main:app --reload
```

---

## API Workflow

```
Frontend
     │
     ▼
Upload Image
     │
     ▼
FastAPI
     │
     ▼
Machine Learning Model
     │
     ▼
Disease Prediction
     │
     ▼
JSON Response
```

---

## Team

**Project Lead**

- Williams Nwachi

---

## License

MIT License
