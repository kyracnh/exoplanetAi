# Docker image name
IMAGE_NAME = hackathon-app
# Container name
CONTAINER_NAME = hackathon-container
# Current directory (absolute path)
PWD := $(shell pwd)

# Build the Docker image
build:
	docker build -t $(IMAGE_NAME) .

# Run the container with volume binding
run:
	docker run -d \
		--name $(CONTAINER_NAME) \
		-p 5000:5000 \
		-v $(PWD):/app \
		-v $(PWD)/Data:/app/Data \
		-v $(PWD)/preprocessed:/app/preprocessed \
		-v $(PWD)/models:/app/models \
		-v $(PWD)/uploads:/app/uploads \
		-v $(PWD)/results:/app/results \
		-v $(PWD)/templates:/app/templates \
		$(IMAGE_NAME)

# Run the container in interactive mode with volume binding
run-interactive:
	docker run -it \
		--name $(CONTAINER_NAME) \
		-p 5000:5000 \
		-v $(PWD):/app \
		-v $(PWD)/Data:/app/Data \
		-v $(PWD)/preprocessed:/app/preprocessed \
		-v $(PWD)/models:/app/models \
		-v $(PWD)/uploads:/app/uploads \
		-v $(PWD)/results:/app/results \
		-v $(PWD)/templates:/app/templates \
		$(IMAGE_NAME) /bin/bash

# Preprocess data inside container
preprocess:
	docker exec -it $(CONTAINER_NAME) python data_loader.py

# Train model inside container
train:
	docker exec -it $(CONTAINER_NAME) python model_trainer.py

# Run the Flask app inside the container
run-app:
	docker exec -it $(CONTAINER_NAME) python app.py

# Access container shell
shell:
	docker exec -it $(CONTAINER_NAME) /bin/bash

# View logs
logs:
	docker logs -f $(CONTAINER_NAME)

# Stop the container
stop:
	docker stop $(CONTAINER_NAME)

# Start the container
start:
	docker start $(CONTAINER_NAME)

# Restart the container
restart: stop start

# Remove the container
rm:
	docker rm $(CONTAINER_NAME)

# Remove the container forcefully
rm-force:
	docker rm -f $(CONTAINER_NAME)

# Clean everything (container + image)
clean: stop rm
	docker image prune -a -f

# Full cleanup (remove container, image, and dangling volumes)
clean-all: rm-force
	docker rmi $(IMAGE_NAME) -f
	docker volume prune -f
	docker image prune -a -f

# Setup: build and run with volume binding
setup: build run

# Full workflow: setup, preprocess, train
full-setup: setup preprocess train

# Help command
help:
	@echo "Available commands:"
	@echo "  make build           - Build the Docker image"
	@echo "  make run             - Run container in detached mode with volume binding"
	@echo "  make run-interactive - Run container in interactive mode"
	@echo "  make preprocess      - Run data preprocessing"
	@echo "  make train           - Train the model"
	@echo "  make run-app         - Run Flask app"
	@echo "  make shell           - Access container shell"
	@echo "  make logs            - View container logs"
	@echo "  make stop            - Stop the container"
	@echo "  make start           - Start the container"
	@echo "  make restart         - Restart the container"
	@echo "  make rm              - Remove the container"
	@echo "  make clean           - Stop and remove container, prune images"
	@echo "  make clean-all       - Complete cleanup"
	@echo "  make setup           - Build and run (recommended for first time)"
	@echo "  make full-setup      - Build, run, preprocess, and train"