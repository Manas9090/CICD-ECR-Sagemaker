# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI()

# Define input data model for prediction
class PredictionInput(BaseModel):
    inputs: list[list[float]]

# Load your trained model (.pkl) at startup
MODEL_PATH = "/opt/program/model.pkl"  # make sure your model is here
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None
    print("Warning: model.pkl not found, using dummy predictions.")

# Health check endpoint
@app.get("/ping")
def ping():
    return {"status": "ok"}

# Inference endpoint
@app.post("/invocations")
def invocations(payload: PredictionInput):
    try:
        data = np.array(payload.inputs)
        if model:
            predictions = model.predict(data).tolist()
        else:
            # dummy output if model is missing
            predictions = ["dummy_output" for _ in range(len(data))]
        return {"predictions": predictions, "received": payload.inputs}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
