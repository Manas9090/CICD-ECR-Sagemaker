FROM python:3.10-slim
WORKDIR /app

# Copy project files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Make serve script executable
RUN chmod +x serve

# Default command for SageMaker
ENTRYPOINT ["./serve"]
