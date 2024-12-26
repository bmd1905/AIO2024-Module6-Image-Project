.PHONY: init deploy client clean streamlit

clean:
	ray stop || true
	rm -rf /tmp/ray_project/*

init: clean
	mkdir -p /tmp/ray_project
	ray start --head --disable-usage-stats --temp-dir=/tmp/ray_project

deploy:
	@echo "Starting object detection server..."
	serve run object_detection:entrypoint --address="auto"

deploy_ocr:
	@echo "Starting OCR server..."
	serve run ocr:entrypoint --address="auto"

streamlit:
	@echo "Starting Streamlit app..."
	streamlit run app.py
