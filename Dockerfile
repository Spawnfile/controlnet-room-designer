# Dockerfile for building a Docker image hosting the image-to-image generation server-side application
# Select an official Python 3.9 runtime image as foundation
FROM python:3.9

# Define a logical workspace directory within the container
WORKDIR /code 

# Copy the curated requirements.txt manifest listing all dependencies required by the application
COPY ./server/requirements.txt /code/requirements.txt

# Install packaged software prerequisites
RUN pip install --upgrade --no-cache-dir -r /code/requirements.txt

# Install packaged software prerequisites
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Transfer the fully qualified server application code tree onto the container filesystem
COPY ./server /code/app

# Declare a startup script invoking UVICORN application server, exposing the public interface, attaching to port 80, and activating development mode
CMD ["uvicorn", "app.server_side:app", "--host", "0.0.0.0", "--port", "80", "--reload"]
