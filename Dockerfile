# Use official Python image
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Copy project files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port for inference
EXPOSE 8080

# Command to run inference (adjust if using serve script)
CMD ["python", "inference.py"]
