from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from google.cloud import firestore
import tensorflow as tf
import numpy as np
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "keyfile.json"

# Inisialisasi Firestore Client
db = firestore.Client()

# Load TensorFlow Model (.h5)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "model1.h5")
model = tf.keras.models.load_model(MODEL_PATH)

# FastAPI Instance
app = FastAPI()

# Skema Request Body
class HealthData(BaseModel):
    weight: float
    height: float
    age: int
    gender: str

# Preprocessing Data
def preprocess_data(data):
    # Map Gender to Numeric
    gender_map = {"Male": 0, "Female": 1}
    gender = gender_map.get(data["gender"], 0)
    bmi = data["weight"] / (data["height"] / 100) ** 2
    input_features = np.array([[gender, bmi, data["age"]]])
    return input_features, bmi

# Simpan Data ke Firestore
def save_to_firestore(data):
    db.collection("health_data").add(data)

# Prediksi dan Perhitungan
def calculate_recommendations(input_features):
    prediction = model.predict(input_features)
    recommended_steps = int(np.round(prediction[0][0] / 100) * 100)
    return recommended_steps

# Endpoint Tunggal
@app.post("/process-data")
async def process_data(data: HealthData, background_tasks: BackgroundTasks):
    try:
        # Preprocess Data
        input_features, bmi = preprocess_data(data.dict())

        # Simpan ke Firestore
        background_tasks.add_task(save_to_firestore, data.dict())

        # Prediksi Model
        recommended_steps = calculate_recommendations(input_features)

        # Response
        response = {
            "success": True,
            "message": "Data processed successfully",
            "data": {
                "gender": data.gender,
                "age": data.age,
                "height": data.height,
                "weight": data.weight,
                "bmi": round(bmi, 2),
                "recommended_steps": recommended_steps,
            },
        }
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
