import base64

import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from io import BytesIO
from PIL import Image, ImageEnhance
from starlette.middleware.cors import CORSMiddleware
import tensorflow as tf

model = tf.keras.models.load_model('mnist_cnn_model.keras')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the mapping of class indices to digits
class_mapping = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}


class ImageData(BaseModel):
    image: str

def truncate_probabilities(probabilities, decimals):
    """Truncate probabilities to the specified number of decimal places."""
    factor = 10.0 ** decimals
    return np.floor(probabilities * factor) / factor


@app.post("/inference")
async def do_inference(data: ImageData):
    # Decode the Base64 image
    image_data = data.image.split(",")[1]  # Remove data:image/png;base64 prefix
    image_bytes = base64.b64decode(image_data)
    image = Image.open(BytesIO(image_bytes))

    if image.mode == 'RGBA':
        image = image.convert('RGB')

    image = image.resize((28, 28)).convert('L')

    image = ImageEnhance.Contrast(image)
    image = image.enhance(2.0)  # Increase contrast by a factor of 2

    # Convert to NumPy array and normalize
    image_array = np.array(image) / 255.0

    # Add batch dimension (1, 28, 28, 1) for model prediction
    image_array = image_array.reshape(1, 28, 28, 1)

    # Run the model inference
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_digit = class_mapping[predicted_class]
    print(predicted_digit)

    # Extract probabilities for the predicted class
    probabilities = predictions[0].tolist()
    probabilities = [round(p, 2) for p in probabilities]
    print(probabilities)

    # Return both predicted digit and probabilities
    return {
        "predicted_digit": predicted_digit,
        "probabilities": probabilities
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
