# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 22:28:28 2021

@author: dell
"""

# Import packages
import cv2
import mediapipe as mp
import pyautogui as pg
from pynput.mouse import Listener

# mp_holistic = mp.solutions.holistic # Holistic model
# mp_drawing = mp.solutions.drawing_utils # Drawing utilities


mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# def mediapipe_detection(image, model):
# 	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
# 	image.flags.writeable = False				 # Image is no longer writable
# 	results = model.process(image)				 # Make prediction
# 	image.flags.writeable = True				 # Image is now writable
# 	image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR CONVERSION RGB 2 BGR
# 	return image, results

# def draw_landmarks(image, results):
# 	mp_drawing.draw_landmarks(
# 	image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION) # Draw face connections
# 	mp_drawing.draw_landmarks(
# 	image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
# 	mp_drawing.draw_landmarks(
# 	image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
# 	mp_drawing.draw_landmarks(
# 	image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections
# 	
# def draw_styled_landmarks(image, results):
# 	# Draw face connections
# 	mp_drawing.draw_landmarks(
# 	image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
# 	mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
# 	mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))
# 	# Draw pose connections
# 	mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
# 							mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
# 							mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
# 							)
# 	# Draw left hand connections
# 	mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
# 							mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
# 							mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
# 							)
# 	# Draw right hand connections
# 	mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
# 							mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
# 							mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
# 							)
# Main function
cap = cv2.VideoCapture(0)
# Set mediapipe model
# with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("can't receive frame")
        continue
    # x,y,c=frame.shape
    frame = cv2.flip(frame, 1)
    # image, results = mediapipe_detection(frame, holistic)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)
    coordinates = []
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
            for lm in handslms.landmark:
                landmarks.append([lm.x, lm.y])
        # print(landmarks[8],'\t',landmarks[12])

        # calculate distance between index and middle finger tip
        dx = landmarks[8][0] - landmarks[12][0]
        dy = landmarks[8][1] - landmarks[12][1]
        dist = (dx ** 2 + dy ** 2) ** 0.5
        print(f"x={landmarks[8][0]} , y={landmarks[8][1]}")
        print("Distance between fingers:", dist)
        coordinates = [[landmarks[8][0], landmarks[8][1]], [landmarks[12][0], landmarks[12][1]]]
        # number 8 is the tip of the index finger
        # 12 is the tip of the middle finger

        cx = pg.size()[0] * landmarks[8][0]
        cy = pg.size()[1] * landmarks[8][1]

        pg.moveTo(cx, cy)
        '''
        if dist<=0.05:
            pg.mouseDown()
        else:
            pg.mouseUp()
        '''
    point_size = 1
    point_color = (0, 255, 0)  # BGR
    thickness = 4  # 0 、4、8


    Display_w, Display_h = 600, 400
    # draw_styled_landmarks(image, results)
    frame = cv2.resize(frame, (Display_w, Display_h))

    for coor in coordinates:
        # print(coor)
        cv2.circle(frame, (int(coor[0] * Display_w), int(coor[1] * Display_h)), point_size, point_color, thickness)

    cv2.imshow('OpenCV Feed', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()

cv2.destroyAllWindows()
