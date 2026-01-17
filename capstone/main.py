from fastapi import FastAPI, UploadFile, File, HTTPException
import tensorflow as tf
from tensorflow.keras.applications.xception import preprocess_input
import numpy as np
from PIL import Image
import io
import json
import uvicorn

app = FastAPI(title="Holden Parker - Capstone - Sports Image Classifier")
model = tf.keras.models.load_model("capstone_model.keras")
with open("classes.json", "r") as f:
    classes = json.load(f)

def classify(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
    X = np.array([preprocess_input(np.array(image))])

    pred = model.predict(X)
    return dict(zip(classes, pred[0]))

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        raise HTTPException(status_code=400, detail="Image must be png or jpg.")
    
    image_bytes = await file.read()
    
    pred = classify(image_bytes)
    result = sorted(pred.items(), key=lambda item: item[1], reverse=True)[:3]
    return {
        "predictions": [
            {"class": c, "confidence": float(p)} for c, p in result
        ]
    }

def main():
    print("Hello from capstone!")
    uvicorn.run(app, host="0.0.0.0", port=9697)

if __name__ == "__main__":
    main()
