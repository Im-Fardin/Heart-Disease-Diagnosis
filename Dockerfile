FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into the container
COPY . .

# Create models directory to avoid save errors
RUN mkdir -p models

# Run the pipeline!
CMD ["python", "-u", "src/train.py"]
