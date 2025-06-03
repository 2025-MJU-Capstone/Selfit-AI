import mediapipe as mp
import math
import cv2

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import Final

class poseLandmark:
    def __init__(self, image, gender):
        self.image = image
        self.gender = gender
        self.pose = mp.solutions.pose.Pose(static_image_mode=True)
        self.results = self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    @staticmethod
    def distance3d(a, b):
        return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2 + (a.z - b.z)**2)
    
    @staticmethod
    def ellipse_circumference(a, b):
        return math.pi * math.sqrt(2 * (a**2 + b**2))
    
    @staticmethod
    def estimate_circumference(width_cm, depth_factor):
        a = width_cm / 2
        b = (width_cm * depth_factor) / 2
        return poseLandmark.ellipse_circumference(a, b)
    
    @staticmethod
    def average_distance3d(pairs):
        return (sum(poseLandmark.distance3d(a, b) for a, b in pairs) / len(pairs)) * 100

    @staticmethod
    def analyze_body(gender, image_path):
        modelPath: Final = '/Users/judohyeon/workspace/Selfit_AI/pose/pose_landmarker_full.task'
        imagePath: Final = image_path
        image = mp.Image.create_from_file(imagePath)

        baseOptions = mp.tasks.BaseOptions
        poseLandMarker = mp.tasks.vision.PoseLandmarker
        poseLandMarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        visionRunningMode = mp.tasks.vision.RunningMode

        options = poseLandMarkerOptions(
            base_options = baseOptions(model_asset_path=modelPath),
            output_segmentation_masks=False,
            running_mode=visionRunningMode.IMAGE)

        with poseLandMarker.create_from_options(options) as landmarker:
            results = landmarker.detect(image)
    
            if results.pose_world_landmarks:
                print("포즈 인식 완료했습니다.")
            else:
                print("포즈를 인식하지 못했습니다.")
    
            lm = results.pose_world_landmarks[0]
    
            left_leg = poseLandmark.distance3d(lm[23], lm[25]) * 100 + poseLandmark.distance3d(lm[25], lm[27]) * 100
            right_leg = poseLandmark.distance3d(lm[24], lm[26]) * 100 + poseLandmark.distance3d(lm[26], lm[28]) * 100

            shoulder_width = poseLandmark.average_distance3d([(lm[11], lm[12]), (lm[13], lm[14])])

            hip_width = poseLandmark.distance3d(lm[23], lm[24]) * 100

            left_arm = poseLandmark.distance3d(lm[11], lm[13]) * 100 + poseLandmark.distance3d(lm[13], lm[15]) * 100
            right_arm = poseLandmark.distance3d(lm[12], lm[14]) * 100 + poseLandmark.distance3d(lm[14], lm[16]) * 100
    
            # 남성 기준 보정치
            male_chest_circum = poseLandmark.estimate_circumference(shoulder_width, 0.6)
            male_waist_circum = poseLandmark.estimate_circumference(hip_width, 1.3)

            # 여성 기준 보정치
            female_chest_circum = poseLandmark.estimate_circumference(shoulder_width, 0.9)
            female_waist_circum = poseLandmark.estimate_circumference(hip_width, 1.2)
        
            if (gender == 'male'):
                body_measurements = {
                    "left_leg_cm": float(f"{left_leg:.1f}"),
                    "right_leg_cm": float(f"{right_leg:.1f}"),
                    "shoulder_width_cm":  float(f"{shoulder_width:.1f}"),
                    "hip_width_cm":  float(f"{hip_width:.1f}"),
                    "left_arm_cm":  float(f"{left_arm:.1f}"),
                    "right_arm_cm":  float(f"{right_arm:.1f}"),
                    "chest_circum_cm": float(f"{male_chest_circum:.1f}"),
                    "waist_circum_cm": float(f"{male_waist_circum:.1f}")
                }
            else :
                body_measurements = {
                    "left_leg_cm": float(f"{left_leg:.1f}"),
                    "right_leg_cm": float(f"{right_leg:.1f}"),
                    "shoulder_width_cm":  float(f"{shoulder_width:.1f}"),
                    "hip_width_cm":  float(f"{hip_width:.1f}"),
                    "left_arm_cm":  float(f"{left_arm:.1f}"),
                    "right_arm_cm":  float(f"{right_arm:.1f}"),
                    "chest_circum_cm": float(f"{female_chest_circum:.1f}"),
                    "waist_circum_cm": float(f"{female_waist_circum:.1f}")
                }
            
            return body_measurements
