# AIO-2024 Module 6 - Image Project


https://github.com/user-attachments/assets/e1415720-2ea6-45af-8c38-a393db90e4d2


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
```

## Start deployment

```bash
cd deployment
make deploy_ocr
```

## Start the Streamlit App

```bash
cd deployment
make streamlit
```

You can:

- Go to `localhost:8265` to see the dashboard of the server
- Access the Swagger UI at `localhost:8000/docs`
- The UI is avaiable at `localhost:8501`
