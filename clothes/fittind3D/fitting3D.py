import requests
import time

MESHY_API_KEY = "msy_10PceQF6qxRbq2GSLngncq4r4ypGSrC8B8cR"  
MESHY_API_URL = "https://api.meshy.ai/openapi/v1/image-to-3d"
HEADERS = {
    "Authorization": f"Bearer {MESHY_API_KEY}"
}

def generate_3d_model_from_image(image_url: str, poll_interval: int = 10, timeout: int = 600) -> dict:
    payload = {
        "image_url": image_url,
        "enable_pbr": False,
        "should_remesh": True,
        "should_texture": True
    }

    response = requests.post(MESHY_API_URL, headers=HEADERS, json=payload)
    print("Meshy API raw response:", response.status_code, response.text)
    response.raise_for_status()
    job_id = response.json().get("result")

    if not job_id:
        raise Exception("Failed to get job_id from Meshy API")

    print(f"[Meshy] Job started: {job_id}")

    # Step 2: Polling for completion
    status_url = f"{MESHY_API_URL}/{job_id}"
    start_time = time.time()

    while True:
        status_response = requests.get(status_url, headers=HEADERS)
        status_response.raise_for_status()
        result = status_response.json()

        if result["status"] == "SUCCEEDED":
            print(f"[Meshy] Job completed: {job_id}")
            return {
                "model_url": result["model_url"]
            }
        elif result["status"] == "failed":
            raise Exception(f"Meshy job failed: {job_id}")
        
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Timeout waiting for Meshy job: {job_id}")

        print(f"[Meshy] Job still processing... waiting {poll_interval}s")
        time.sleep(poll_interval)
