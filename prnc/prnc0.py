import time
#系统客户端
import win32com.client
dehua = win32com.client.Dispatch("SAPI.SPVOICE")

while 1:
    dehua.Speak("sunck is a handsome man")
    time.sleep(5)
