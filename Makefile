# Build the Docker image
build:
	docker build -t exoplanet-app .

# Run the container interactively (you can access bash inside)
bash:
	docker run -it --rm -v $(PWD):/app exoplanet-app bash

# Run the Flask app inside the container
run:
	docker run -it --rm -p 5000:5000 -v $(PWD):/app exoplanet-app python3 app.py
