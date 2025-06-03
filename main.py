from fastapi import FastAPI, Query
from pose import poseLandmark
from clothes.ftting2D import fitting
from clothes.fittind3D import fitting3D

import requests
import cv2
import numpy as np
import mediapipe as mp

app = FastAPI()

@app.get("/body")
async def analyze_body(gender: str = Query(...), image_url: str = Query(...)):
    response = requests.get(image_url)
    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    
    result = poseLandmark.analyze_body(gender, image)
    
    return result

@app.get("/fitting/3D")
async def analyze_body(body_image_url: str = Query(...), clothes_image_url: str = Query(...)):
    image_2d = fitting.run(body_image_url, clothes_image_url)
    result = fitting3D.generate_3d_model_from_image(image_2d)
    result["image_2d"] = image_2d
    
    return result

@app.get("/body/3D")
async def analyze_body(body_image_url: str = Query(...)):
    result = fitting3D.generate_3d_model_from_image(body_image_url)
    
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)