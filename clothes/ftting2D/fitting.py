import os
import time
import requests
import base64
 
class fitting:
    API_KEY = "fa-3sE1CTrqYa4Q-7smRsl4bqJVzNOp9AfBpJMWe"
    BASE_URL = "https://api.fashn.ai/v1"
 
    @classmethod
    def run(cls, model_image: str, garment_image: str):
        input_data = {
            "model_image": model_image,
            "garment_image": garment_image,
            "category": "auto",
            "mode": "quality"
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cls.API_KEY}"
        }

        # Step 1: POST to /run
        run_response = requests.post(f"{cls.BASE_URL}/run", json=input_data, headers=headers)
        if run_response.status_code != 200:
            return {"error": "Fashn run API failed", "detail": run_response.text}

        prediction_id = run_response.json().get("id")
        if not prediction_id:
            return {"error": "No prediction ID returned from Fashn"}

        # Step 2: Poll until completed
        while True:
            status_response = requests.get(f"{cls.BASE_URL}/status/{prediction_id}", headers=headers)
            if status_response.status_code != 200:
                return {"error": "Status check failed", "detail": status_response.text}

            status_data = status_response.json()
            status = status_data.get("status")

            if status == "completed":
                return {"result": status_data.get("output")}
            elif status in ["starting", "in_queue", "processing"]:
                time.sleep(3)
            else:
                return {"error": "Prediction failed", "detail": status_data.get("error")}
