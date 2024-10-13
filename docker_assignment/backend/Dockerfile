# Use the official Python image from the Docker Hub
FROM python:3.10-slim

# Set environment variables
#Purpose: This variable disables Python's output buffering.
#Effect: By setting this, you ensure that the output from Python is sent directly to the terminal or log files. This is particularly useful in a Docker environment where you want real-time logs.
ENV PYTHONUNBUFFERED=1
#Purpose: This variable prevents Python from writing .pyc files (compiled bytecode).
#Effect: This can save some disk space and reduce write operations within the container. Since Docker images are layered, avoiding unnecessary files can help keep the image size smaller.
ENV PYTHONDONTWRITEBYTECODE=1

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the FastAPI application with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
