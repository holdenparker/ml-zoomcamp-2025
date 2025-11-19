from fastapi import FastAPI
import uvicorn
import pickle
from typing import Dict, Any

app = FastAPI(title="Holden Parker - Midterm - Predict Salary")

with open('midterm_model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

def predict_single(employee):
    X = dv.transform([employee])
    result = model.predict(X)[0]
    return float(result)

@app.post("/predict")
def predict(employee: Dict[str, Any]):
    pred_salary = predict_single(employee)

    return {
        "predicted_salary": pred_salary
    }

def main():
    print("Hello from midterm!")
    uvicorn.run(app, host="0.0.0.0", port=9696)

if __name__ == "__main__":
    main()
