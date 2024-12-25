# AIO-2024 Module 6 - Image Project

In this project, we will train a YOLOv11 model to detect

## Setup Env

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Server

```bash
cd deployment
make init
make deploy_ocr
```

You can:

- Go to `localhost:8265` to see the dashboard of the server
- Access the Swagger UI at `localhost:8000/docs`
