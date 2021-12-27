import ctypes
import struct
import numpy as np
import cv2
import mediapipe as mp
import pyautogui as pg

# define C struct
c_int_p = ctypes.POINTER(ctypes.c_int)


class CLS_Gesture(ctypes.Structure):
    _fields_ = [("m_Gesture_Recognition_Result", c_int_p), ("m_HandUp_HandDown_Detect_Result", c_int_p)]


# load dll
dll = ctypes.windll.LoadLibrary('./Mediapipe_Hand_Tracking.dll')

# get function
Func_init = dll.Mediapipe_Hand_Tracking_Init
Func_init.argtypes = [ctypes.c_char_p]
Func_init.restypes = ctypes.c_int

Func_Track_Face = dll.Mediapipe_Hand_Tracking_Detect_Frame_Direct
Func_Track_Face.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, CLS_Gesture]
Func_Track_Face.restypes = ctypes.c_int

# init the graph
model_path = './hand_tracking_desktop_live.pbtxt'
pStr = ctypes.c_char_p()
pStr.value = model_path.encode("utf-8")
nRst = Func_init(pStr)
print("Graph init status:", nRst)

Gest = CLS_Gesture()

# cap = cv2.VideoCapture(0)

while 1:#cap.isOpened():
    # ret, frame = cap.read()
    # if not ret:
    #     print("can't receive frame")
    #     continue
    # y, x, c = frame.shape
    # frame = cv2.flip(frame, 1)
    # framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # framergb = np.ascontiguousarray(framergb, dtype=np.uint8)  # 如果不是C连续的内存，必须强制转换
    # data_p = framergb.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
    ##frame_ctypes_ptr = ctypes.cast(framergb.ctypes.data, ctypes.POINTER(ctypes.c_int16))
    # cv2.imshow('OpenCV Feed', framergb)
    rows, cols=100,100
    ret_img = np.zeros(dtype=np.uint8, shape=(rows, cols, 3))
    ret_img = np.ascontiguousarray(ret_img)
    nRst = Func_Track_Face(rows, cols, ret_img.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)), Gest)
    # cv2.imshow('OpenCV Feed-after', framergb)
    print(nRst)


    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
