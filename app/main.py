from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import pymysql
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
import os
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO)

# Load Environment Variables
load_dotenv()

# Database Configuration
db_config = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
    "port": int(os.getenv("DB_PORT", 3306)),  # Default ke 3306 jika tidak ada
}

def connect_to_db():
    """Create a connection to MySQL database."""
    return pymysql.connect(**db_config)

# Load TensorFlow Model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "model1.h5")
model = tf.keras.models.load_model(MODEL_PATH)

# Initialize Scalers
scaler_X = StandardScaler()
scaler_X.mean_ = np.array([0.26, 23.86, 31.12])  # Mean dari gender, BMI, dan age
scaler_X.scale_ = np.array([0.44, 5.15, 8.85])  # Standard deviasi dari gender, BMI, dan age

scaler_y = StandardScaler()
scaler_y.mean_ = np.array([3755.55])  # Mean dari daily_steps
scaler_y.scale_ = np.array([971.24])  # Standard deviasi dari daily_steps

# Initialize FastAPI
app = FastAPI()

# Request Body Schema
class HealthData(BaseModel):
    user_id: int
    weight: float
    height: float
    age: int
    gender: str

# Data Preprocessing Function
def preprocess_data(data):
    """Preprocess input data for prediction."""
    gender_map = {"Male": 0, "Female": 1}
    gender = gender_map.get(data["gender"], 0)
    bmi = data["weight"] / (data["height"] / 100) ** 2

    input_features = np.array([[gender, bmi, data["age"]]])
    input_features_scaled = scaler_X.transform(input_features)
    logging.info(f"Preprocessed Features (Scaled): {input_features_scaled}")
    return input_features_scaled, bmi

# Save Data to MySQL
def save_to_mysql(user_id, data):
    """Save processed data to MySQL."""
    try:
        connection = connect_to_db()
        with connection.cursor() as cursor:
            sql = """
            INSERT INTO predictions (user_id, gender, age, height, weight, bmi, recommended_steps)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(sql, (
                user_id,
                data["gender"],
                data["age"],
                data["height"],
                data["weight"],
                data["bmi"],
                data["recommended_steps"],
            ))
        connection.commit()
        logging.info("Data berhasil disimpan ke MySQL.")
    except Exception as e:
        logging.error(f"Error saat menyimpan ke MySQL: {e}")
    finally:
        connection.close()

# Calculate Recommendations
def calculate_recommendations(input_features):
    """Calculate recommended steps using the model."""
    prediction_scaled = model.predict(input_features)
    prediction = scaler_y.inverse_transform(prediction_scaled)
    recommended_steps = int(np.round(prediction[0][0] / 100) * 100)
    logging.info(f"Predicted Steps: {recommended_steps}")
    return recommended_steps

# API Endpoint
@app.post("/process-data")
async def process_data(data: HealthData):
    """Process input data, make predictions, and save to database."""
    try:
        # Preprocess Data
        input_features, bmi = preprocess_data(data.dict())

        # Calculate Recommendations
        recommended_steps = calculate_recommendations(input_features)

        # Save Data to MySQL
        save_to_mysql(data.user_id, {
            "gender": data.gender,
            "age": data.age,
            "height": data.height,
            "weight": data.weight,
            "bmi": round(bmi, 2),
            "recommended_steps": recommended_steps,
        })

        # Response
        response = {
            "success": True,
            "message": "Data processed successfully",
            "data": {
                "user_id": data.user_id,
                "gender": data.gender,
                "age": data.age,
                "height": data.height,
                "weight": data.weight,
                "bmi": round(bmi, 2),
                "recommended_steps": recommended_steps,
            },
        }
        logging.info(f"Response: {response}")
        return response

    except Exception as e:
        logging.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
