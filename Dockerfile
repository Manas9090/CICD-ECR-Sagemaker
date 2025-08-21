FROM python:3.9-slim 

RUN pip install --no-cache-dir flask joblib scikit-learn 

# Copy model and inference code 
COPY model /opt/ml/model/
COPY inference.py /opt/ml/code/inference.py 

EXPOSE 8080

ENTRYPOINT ["python", "/opt/ml/code/inference.py"]
