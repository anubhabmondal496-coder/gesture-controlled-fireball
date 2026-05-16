import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import random
import time

def overlay_fire(background, sprite, center, size):
    """Overlays a transparent sprite with 'Additive Blending' for a glow effect."""
    x, y = center
    res = int(size * 1.5) # Scale the sprite based on hand distance
    if res <= 0: return background
    
    # Resize sprite
    sprite_resized = cv2.resize(sprite, (res, res))
    
    # Calculate ROI coordinates
    x1, y1 = max(0, x - res//2), max(0, y - res//2)
    x2, y2 = min(background.shape[1], x + res//2), min(background.shape[0], y + res//2)
    
    # Adjust sprite size if it hits screen edges
    sprite_part = sprite_resized[0:(y2-y1), 0:(x2-x1)]
    
    # Logic: Additive Blending (Brings out the reds and oranges)
    background[y1:y2, x1:x2] = cv2.addWeighted(background[y1:y2, x1:x2], 1.0, sprite_part, 0.8, 0)
    return background


base = python.BaseOptions(model_asset_path= "hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options= base,
    min_tracking_confidence= 0.6,
    min_hand_detection_confidence= 0.6,
    min_hand_presence_confidence= 0.6,
    num_hands= 2
)

detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
fire_cap = cv2.VideoCapture("images_videos/fireball.mov")

while cap.isOpened():
    r,fps = cap.read()
    if r == True:
        fps = cv2.flip(fps,1)
        fps = cv2.resize(fps,(700,700))
        rgb_fps = cv2.cvtColor(fps,cv2.COLOR_BGR2RGB)
        mp_fps = mp.Image(mp.ImageFormat.SRGB,data= rgb_fps)

        detection_result = detector.detect(mp_fps)
        if detection_result.hand_landmarks:
            for landmark in detection_result.hand_landmarks:
                for landmarks in landmark:
                    x = int(landmarks.x * fps.shape[1])
                    y = int(landmarks.y * fps.shape[0])
                    cv2.circle(fps,(x,y),5,(0,0,255),-1)
                connections = mp.solutions.hands.HAND_CONNECTIONS
                for connection in connections:
                    start = connection[0]
                    end = connection[1]

                    start_index = landmark[start]
                    end_index = landmark[end]

                    pt1 = (int(start_index.x * fps.shape[1]), int(start_index.y * fps.shape[0]))
                    pt2 = (int(end_index.x * fps.shape[1]), int(end_index.y * fps.shape[0]))

                    cv2.line(fps,pt1,pt2,(255,255,255),2)

            if len(detection_result.hand_landmarks) == 2:
                hand1 = detection_result.hand_landmarks[0]
                hand2 = detection_result.hand_landmarks[1]

                h,w,t = fps.shape

                x1,y1 = int(hand1[0].x * w),int(hand1[0].y * h)
                x2,y2 = int(hand2[10].x * w),int(hand2[10].y * h)

                distance = int(math.sqrt(pow(x1 - x2,2) + pow(y1 - y2,2)))
                mid_x,mid_y = int((x1 + x2) // 2) , int((y1 + y2) // 2)

                
                ret_fire, fire_frame = fire_cap.read()
                if not ret_fire:
                    
                    fire_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret_fire, fire_frame = fire_cap.read()
                
                if ret_fire:
                    fps = overlay_fire(fps, fire_frame, (mid_x, mid_y), distance)
        cv2.imshow("window",fps)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    else:
        break
detector.close()
cap.release()
fire_cap.release()
cv2.destroyAllWindows()