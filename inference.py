import pickle
import os

MODEL_PATH = os.path.join("model", "model.pkl")

def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model

def predict(input_data):
    model = load_model()
    return model.predict([input_data])

if __name__ == "__main__":
    sample_input = [1, 2, 3]  # replace with actual input
    print("Prediction:", predict(sample_input))
