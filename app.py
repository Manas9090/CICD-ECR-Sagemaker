# inference.py
from flask import Flask, request, jsonify

app = Flask(__name__)

# Health check
@app.route("/ping", methods=["GET"])
def ping():
    return "OK", 200

# Inference
@app.route("/invocations", methods=["POST"])
def invocations():
    try:
        payload = request.get_json(force=True, silent=True)
        # TODO: replace with your real inference logic
        # Example: echo back the payload
        result = {"received": payload, "prediction": "dummy_output"}
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Local run (optional)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)