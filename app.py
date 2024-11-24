# from tensorflow import keras
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as tf_text  # Necessary for loading BERT models
import tf_keras as k3
from colorama import Fore, Style
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

# from tensorflow.keras.models import load_model

app = FastAPI()

model = k3.models.load_model("detect_LLM_colab_method2")


# Initialize FastAPI
app = FastAPI()

origins = [
    "http://localhost:3000",  # your development web app URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextInput(BaseModel):
    text: str


def preprocess_text(text: str):
    processed_text = text.lower().strip()

    return np.array([processed_text])


def predict(text: str):
    input_data = preprocess_text(text)
    prediction = model.predict(input_data)

    # Check and handle the type of prediction output
    if isinstance(prediction, (np.ndarray, list)):
        output_value = prediction[0]
    elif isinstance(prediction, (np.float32, float)):
        output_value = prediction
    else:
        raise ValueError("Unexpected prediction output type.")

    if isinstance(output_value, np.ndarray):
        output_value = output_value.item()  # Convert single-element array to scalar

    output = round(float(output_value) * 100, 2)
    return output



@app.post("/predict/")
async def make_prediction(input: TextInput):
    try:
        result = predict(input.text)
        return {"prediction": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
