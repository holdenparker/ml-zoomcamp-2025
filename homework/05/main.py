from fastapi import FastAPI
import uvicorn
import pickle
from typing import Dict, Any

app = FastAPI(title="Homework 5 - Lead Scoring")

with open('pipeline_v1.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

def predict_single(customer):
    result = pipeline.predict_proba(customer)[0, 1]
    return float(result)

@app.post("/predict")
def predict(customer: Dict[str, Any]):
    prob = predict_single(customer)

    return {
        "churn_probability": prob,
        "churn": bool(prob >= 0.5)
    }

def main():
    print("Hello from 05!")
    uvicorn.run(app, host="0.0.0.0", port=9696)

if __name__ == "__main__":
    main()
