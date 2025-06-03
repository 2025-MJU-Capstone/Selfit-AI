import os
import tempfile
from fastapi import FastAPI, Query
from pose.poseLandmark import poseLandmark
from clothes.ftting2D import fitting 
from clothes.fittind3D import fitting3D

import requests
import cv2
import numpy as np
import mediapipe as mp

app = FastAPI()

@app.get("/body")
async def analyze_body(gender: str = Query(...), image_url: str = Query(...)):
   # 1. 이미지 다운로드
    response = requests.get(image_url)

    # 2. 임시 파일에 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        tmp_file.write(response.content)
        tmp_file_path = tmp_file.name

    try:
        # 3. 임시 파일 경로를 넘겨줌
        result = poseLandmark.analyze_body(gender, tmp_file_path)
    finally:
        # 4. 임시 파일 삭제
        os.remove(tmp_file_path)

    return result

@app.get("/fitting/3D")
async def analyze_body(body_image_url: str = Query(...), clothes_image_url: str = Query(...)):
    fitting_result = fitting.fitting2D.run(body_image_url, clothes_image_url)
    image_url = fitting_result["result"][0]  # 첫 번째 URL 추출

    result = fitting3D.generate_3d_model_from_image(image_url)
    result["image_2d"] = image_url
    
    return result

@app.get("/body/3D")
async def analyze_body(body_image_url: str = Query(...)):
    result = fitting3D.generate_3d_model_from_image(body_image_url)
    
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)