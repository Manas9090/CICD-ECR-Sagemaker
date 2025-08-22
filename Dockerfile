# Use AWS public Python image to avoid Docker Hub pull limits
FROM public.ecr.aws/sam/build-python3.9:latest

# Set working directory SageMaker BYOC convention
WORKDIR /opt/program

# Copy requirements first for caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into /opt/program
COPY . .

# Put the serve launcher on PATH and make it executable
RUN cp /opt/program/serve /usr/local/bin/serve && chmod +x /usr/local/bin/serve

# SageMaker expects the container to listen on 8080
EXPOSE 8080

# ENTRYPOINT runs first; SageMaker will invoke the container with "serve"
ENTRYPOINT ["serve"]