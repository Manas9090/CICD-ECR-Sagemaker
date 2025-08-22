# Use AWS public Python image
FROM public.ecr.aws/sam/build-python3.9:latest

# Set working directory (SageMaker BYOC convention)
WORKDIR /opt/program

# Copy requirements first for caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Expose the port SageMaker expects
EXPOSE 8080

# ENTRYPOINT to run the FastAPI server
ENTRYPOINT ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
