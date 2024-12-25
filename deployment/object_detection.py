import os
import tempfile
from io import BytesIO

import numpy as np
import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image
from ray import serve
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

app = FastAPI()


@serve.deployment(num_replicas=1)
@serve.ingress(app)
class APIIngress:
    def __init__(self, object_detection_handle):
        self.handle = object_detection_handle

    async def process_image(self, image_data: bytes) -> Response:
        """Common image processing logic for both URL and file upload"""
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                temp_file.write(image_data)
                temp_file_path = temp_file.name

            # Request detection results using the temp file path
            bboxes, classes, names, confs = await self.handle.detect.remote(
                temp_file_path
            )

            # Open the image from the temp file
            image = Image.open(temp_file_path)
            image_array = np.array(image)

            # Initialize Annotator
            annotator = Annotator(image_array, font="Arial.ttf", pil=False)

            # Draw boxes and labels
            for box, cls, conf in zip(bboxes, classes, confs):
                c = int(cls)
                label = f"{names[c]} {conf:.2f}"
                annotator.box_label(box, label, color=colors(c, True))

            # Convert annotated image back to bytes
            annotated_image = Image.fromarray(annotator.result())
            file_stream = BytesIO()
            annotated_image.save(file_stream, format="PNG")
            file_stream.seek(0)

            # Clean up the temporary file
            os.unlink(temp_file_path)

            return Response(content=file_stream.getvalue(), media_type="image/png")

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing image: {e}")

    @app.get("/detect", response_class=Response)
    async def detect_url(self, image_url: str):
        """Endpoint for processing images from URLs"""
        try:
            response = requests.get(image_url)
            response.raise_for_status()
            return await self.process_image(response.content)
        except requests.RequestException as e:
            raise HTTPException(status_code=400, detail=f"Error downloading image: {e}")

    @app.post("/detect/upload", response_class=Response)
    async def detect_upload(self, file: UploadFile = File(...)):
        """Endpoint for processing uploaded image files"""
        try:
            # Validate file type
            if not file.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail="File must be an image")

            # Read file content
            content = await file.read()
            return await self.process_image(content)

        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error processing uploaded file: {e}"
            )


@serve.deployment(
    ray_actor_options={"num_gpus": 0.2, "num_cpus": 2},
    autoscaling_config={"min_replicas": 1, "max_replicas": 4},
)
class ObjectDetection:
    def __init__(self):
        self.model = YOLO("yolo11n.pt")

    def detect(self, image_path: str):
        try:
            # Perform object detection
            results = self.model(image_path, verbose=False)[0]
            return (
                results.boxes.xyxy.tolist(),
                results.boxes.cls.tolist(),
                results.names,
                results.boxes.conf.tolist(),
            )

        except requests.RequestException as e:
            raise HTTPException(status_code=400, detail=f"Error downloading image: {e}")
        except ValueError as e:
            raise HTTPException(status_code=415, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing image: {e}")


entrypoint = APIIngress.bind(ObjectDetection.bind())
