FROM python:3.10-slim
WORKDIR /app

# Copy all files into container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Make serve script executable
RUN chmod +x ./serve

# SageMaker expects ENTRYPOINT as ./serve
ENTRYPOINT ["./serve"]
