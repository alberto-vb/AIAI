import io

import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from ultralytics import YOLO
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

model = YOLO("yolov8n.pt")
app = FastAPI()


@app.post("/yolo/upload-image/")
async def yolo_upload_image(file: UploadFile = File(...)):
    # Read the uploaded file
    contents = await file.read()

    # Open the image using Pillow
    image = Image.open(io.BytesIO(contents))

    if image.mode == 'RGBA':
        image = image.convert('RGB')

    target_size = (640, 640)
    resized_image = image.resize(target_size)
    image_array = np.array(resized_image)

    # Run the YOLOv8 model on the image
    results = model(image_array)

    person_count = 0
    person_detections = []

    # Loop through the detections to extract bounding boxes, class IDs, and confidence scores
    for result in results:
        for box in result.boxes.data:
            x_min, y_min, x_max, y_max, confidence, class_id = box.cpu().numpy()

            # Check if the class_id corresponds to "person" (class_id == 0 in COCO dataset)
            if int(class_id) == 0:
                person_count += 1
                person_detections.append({
                    "bounding_box": {
                        "x_min": float(x_min),
                        "y_min": float(y_min),
                        "x_max": float(x_max),
                        "y_max": float(y_max),
                    },
                    "confidence": float(confidence)
                })

    # Return the person count and the list of detected people with their bounding boxes and confidence scores
    return JSONResponse(content={
        "person_count": person_count,
        "person_detections": person_detections
    })


@app.post("/azure-ai-model/upload-image/")
async def azure_ai_model_upload_image(file: UploadFile = File(...)):
    # Read the uploaded file
    contents = await file.read()

    # Open the image using Pillow
    image = Image.open(io.BytesIO(contents))

    image = image.convert('RGB') if image.mode == 'RGBA' else image

    # Convert the image to a byte stream
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='JPEG')  # Ensure the format is correct (e.g., JPEG or PNG)
    image_bytes.seek(0)  # Go back to the start of the byte stream

    # Secrets were removed to avoid exposure
    endpoint_url = "<your-endpoint>"
    key = "<your-key>"
    client = ImageAnalysisClient(endpoint=endpoint_url, credential=AzureKeyCredential(key))
    result = client.analyze(image_data=image_bytes, visual_features=[VisualFeatures.PEOPLE])

    people = result['peopleResult']['values']
    confidence_threshold = 0.5
    person_count = 0
    serialized_detections = []

    for person in people:
        if person['confidence'] >= confidence_threshold:
            person_count += 1
            serialized_detections.append(
                {
                    "bounding_box": {
                        "x_min": person['boundingBox']['x'],
                        "y_min": person['boundingBox']['y'],
                        "x_max": person['boundingBox']['x'] + person['boundingBox']['w'],  # Add width to get right x_max
                        "y_max": person['boundingBox']['y'] + person['boundingBox']['h'],  # Add height to get right y_max
                    },
                    "confidence": person['confidence']
                }
            )

    # Count the number of people with confidence >= confidence_threshold
    person_count = sum(1 for person in people if person['confidence'] >= confidence_threshold)

    # Update the person count in the result dictionary
    result['peopleResult']['person_count'] = person_count

    # Return the result
    return JSONResponse(content={
        "person_count": person_count,
        "person_detections": serialized_detections
    })


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
