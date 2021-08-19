import datetime
import random
import smtplib
import webbrowser

import PySimpleGUI as sg
# voice initiation
import pyttsx3

engine = pyttsx3.init('sapi5', True)
engine.setProperty('rate', 225)  # setting up new voice rate
voices = engine.getProperty('voices')  # get system voice pack
engine.setProperty('voice', voices[2].id)  # changing index, changes voices. 2 for male


def speak(msg, msg_show=0):  # popup windows actually
    engine.say(msg)
    if msg_show != 0:
        msg = msg_show
    speak_layout = [[sg.Text(msg)]]
    window = sg.Window('J.A.R.V.I.S.', speak_layout, auto_close=True, auto_close_duration=2)
    event, values = window.read()
    engine.runAndWait()
    # print(msg)


def wiki(content):  # search wikipedia
    wiki_layout = [[sg.Text(content)], [sg.Button('Ok')]]
    window = sg.Window('J.A.R.V.I.S.', wiki_layout)
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == 'Ok':
            break
    window.close()


def wishMe():
    hour = int(datetime.datetime.now().hour)
    if 4 <= hour < 12:
        t = "Good morning"
    elif 12 <= hour < 18:
        t = "Good afternoon"
    else:
        t = "Good evening"
    greetinglist = ["How may I help you?", "How can I be of service today?", "Good to see you again.",
                    "Hope you are well?"]
    speak(t + " sir. " + random.choice(greetinglist))


def sendEmail(to, content):
    server = smtplib.SMTP('smtp.qq.com', 587)
    server.ehlo()
    server.starttls()
    server.login('your_email@example.com', 'your_login_password')
    server.sendmail('your_email@example.com', to, content)
    server.close()


def net(url):
    try:
        webbrowser.open_new_tab(url)
    except:
        webbrowser.open(url)
