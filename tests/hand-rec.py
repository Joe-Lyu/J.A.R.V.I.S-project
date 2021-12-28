# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 22:28:28 2021

@author: dell
"""

# Import packages
import cv2
import mediapipe as mp
import pyautogui as pg
import mouse
# mp_holistic = mp.solutions.holistic # Holistic model
# mp_drawing = mp.solutions.drawing_utils # Drawing utilities
import win32com
import win32con
import win32gui

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils
pg.FAILSAFE=False
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

def video2screenmapping(cx, cy):
    #print(cx,cy)
    sx = pg.size()[0]*cx*2-pg.size()[0]/2
    sy = pg.size()[1]*cy*2-pg.size()[1]/2
    #print(sx,sy,pg.size()[0])
    sx=min(pg.size()[0]-1,max(sx,1))
    sy=min(pg.size()[1]-1,max(sy,1))
    #print(sx,sy,pg.size()[0])
    return (sx,sy)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("can't receive frame")
        continue
    
    
    x,y,c=frame.shape
    
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
                landmarks.append([lm.x, lm.y, lm.z])
        # print(landmarks[8],'\t',landmarks[12])

        f0,f1,f2=4,8,12
        f0_root,f1_root,f2_root = 1,5,9

        '''
        4 for thumb
        8 for index finger
        12 for middle finger
        '''
        #coordinates = [[landmarks[f1][0], landmarks[f1][1]], [landmarks[f2][0], landmarks[f2][1]]]
        coordinates = [[landmarks[8][0], landmarks[8][1]], [landmarks[12][0], landmarks[12][1]]]

        
        #calculate distance between index and middle finger tip
        '''
        dx=landmarks[f1][0]-landmarks[f2][0]
        dy=landmarks[f1][1]-landmarks[f2][1]
        dist=(dx**2+dy**2)
        '''
        
        # dx=landmarks[8][0]-landmarks[12][0]
        # dy=landmarks[8][1]-landmarks[12][1]
        # dz=landmarks[8][2]-landmarks[12][2]
        # dist=(dx**2+dy**2)

        dx = landmarks[f0][0] - landmarks[f1][0]
        dy = landmarks[f0][1] - landmarks[f1][1]
        dz = landmarks[f0][2] - landmarks[f1][2]
        dist_0_1 = (dx ** 2 + dy ** 2)

        dx = landmarks[f0_root][0] - landmarks[f1_root][0]
        dy = landmarks[f0_root][1] - landmarks[f1_root][1]
        dz = landmarks[f0_root][2] - landmarks[f1_root][2]
        dist_0_1_root = (dx ** 2 + dy ** 2)

        dx = landmarks[f2][0] - landmarks[f1][0]
        dy = landmarks[f2][1] - landmarks[f1][1]
        dz = landmarks[f2][2] - landmarks[f1][2]
        dist_1_2 = (dx ** 2 + dy ** 2)

        dx = landmarks[f2_root][0] - landmarks[f1_root][0]
        dy = landmarks[f2_root][1] - landmarks[f1_root][1]
        dz = landmarks[f2_root][2] - landmarks[f1_root][2]
        dist_1_2_root = (dx ** 2 + dy ** 2)


        #'''
        
        # number 8 is the tip of the index finger
        # 12 is the tip of the middle finger

        # mouse pointing to middle point of the two fingers
        '''
        cx = (landmarks[f1][0]+landmarks[f2][0])/2
        cy = (landmarks[f1][1]+landmarks[f2][1])/2
        '''
        cx,cy=(landmarks[8][0]+landmarks[4][0])/2,(landmarks[8][1]+landmarks[4][1])/2
        cursorpos=video2screenmapping(cx, cy)

    
        if dist_0_1<dist_0_1_root/6:
            mouse.hold('left')
            print("Contact")
        else:
            mouse.release('left')

            print("no contact")
        if dist_1_2<dist_1_2_root*1.2:
            mouse.click()
            print("click")
        mouse.move(cursorpos[0],cursorpos[1], absolute=True,duration=0.05)
        

    point_size = 1
    point_color = (0, 255, 0)  # BGR
    thickness = 4  # 0 、4、8


    #Display_w, Display_h = 600, 400
    # draw_styled_landmarks(image, results)
    #frame = cv2.resize(frame, (Display_w, Display_h))

    for coor in coordinates:
        # print(coor)
        cv2.circle(frame, (int(coor[0] * x), int(coor[1] * y)), point_size, point_color, thickness)

    window_name = 'OpenCV Feed'
    cv2.imshow(window_name, frame)
    print("置顶窗口")
    hwnd = win32gui.FindWindow(None, window_name)
    # 窗口需要正常大小且在后台，不能最小化
    win32gui.ShowWindow(hwnd, win32con.SW_SHOWNORMAL)
    win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                          win32con.SWP_NOMOVE | win32con.SWP_NOACTIVATE | win32con.SWP_NOOWNERZORDER | win32con.SWP_SHOWWINDOW | win32con.SWP_NOSIZE)
    # 取消置顶
    # win32gui.SetWindowPos(hwnd, win32.HWND_NOTOPMOST, 0, 0, 0, 0,win32con.SWP_SHOWWINDOW|win32con.SWP_NOSIZE|win32con.SWP_NOMOVE)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()

cv2.destroyAllWindows()
