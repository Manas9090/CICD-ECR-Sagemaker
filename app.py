# inference.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI()

# Health check
@app.get("/ping")
async def ping():
    return {"status": "OK"}

# Inference endpoint
@app.post("/invocations")
async def invocations(request: Request):
    try:
        payload = await request.json()
        # Replace with your real inference logic
        result = {"received": payload, "prediction": "dummy_output"}
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

# Optional local run
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
