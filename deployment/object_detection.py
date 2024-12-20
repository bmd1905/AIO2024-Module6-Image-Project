from io import BytesIO

import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from PIL import Image
from ray import serve
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

app = FastAPI()

# Constants
MODEL_PATH = "../models/yolov11/detect/train/weights/best.pt"


@serve.deployment(num_replicas=1)
@serve.ingress(app)
class APIIngress:
    def __init__(self, object_detection_handle):
        self.handle = object_detection_handle

    @app.get("/detect", response_class=Response)
    async def detect(self, image_url: str):
        try:
            # Request detection results
            bboxes, classes, names, confs = await self.handle.detect.remote(image_url)

            # Load image
            response = requests.get(image_url)
            response.raise_for_status()

            # Convert PIL Image to numpy array for Annotator
            image_bytes = BytesIO(response.content)
            image = Image.open(image_bytes)
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

            return Response(content=file_stream.getvalue(), media_type="image/png")

        except requests.RequestException as e:
            raise HTTPException(status_code=400, detail=f"Error downloading image: {e}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing image: {e}")


@serve.deployment(
    ray_actor_options={"num_gpus": 0.5, "num_cpus": 4},
    autoscaling_config={"min_replicas": 1, "max_replicas": 2},
)
class ObjectDetection:
    def __init__(self):
        self.model = YOLO(MODEL_PATH)

    def detect(self, image_url: str):
        try:
            # Perform object detection
            results = self.model(image_url, verbose=False)[0]
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
