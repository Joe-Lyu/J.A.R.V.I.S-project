# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 23:37:34 2021

@author: DELL
"""
# =============================================================================
# JARVIS V1.0.2 BETA
# features: 
#                   voice control;
#                   open several websites;
#                   play music (suffle only);
#                   tell time;
#                   send email
# issues:
#                   slow recognition time;
#                   limited abilities & responses;
#                   limited understanding
#                   no conversational skills
#FIXED(1.0.1)       proxy problems
#FIXED(1.0.2)       web browser problems
# =============================================================================
# =============================================================================
# fixed proxys
# minor change to 'quit program'
#
# =============================================================================
import speech_recognition as sr 
import datetime
import wikipedia 
import webbrowser
import os
import sys
import smtplib
import random
import PySimpleGUI as sg

proxy = "http://127.0.0.1:15732"
os.environ['http_proxy'] = proxy 
os.environ['HTTP_PROXY'] = proxy
os.environ['https_proxy'] = proxy
os.environ['HTTPS_PROXY'] = proxy



sg.theme('DarkAmber')   # Add a touch of color
# All the stuff inside your window.
email_layout = [  [sg.Text('J.A.R.V.I.S. email service')],
            [sg.Text('Email content'), sg.InputText()],
            [sg.Text('send to'),sg.InputText()],
            [sg.Button('Ok'), sg.Button('Cancel')] ]
music_layout = [  [sg.Text('J.A.R.V.I.S. music player')],
                [sg.Button('Shuffle music'),sg.Button('Cancel')]]
chrome_path="C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe"
webbrowser.register('chrome', webbrowser.BackgroundBrowser(chrome_path),1)
webbrowser.get('chrome')


def speak(msg):
    speak_layout = [ [sg.Text(msg)]]
    window=sg.Window('J.A.R.V.I.S.', speak_layout,auto_close=True,auto_close_duration=2)
    event, values = window.read()
    #window.close()
    



def wishMe():
    hour = int(datetime.datetime.now().hour)
    if hour>=4 and hour<12:
        t="Good morning"

    elif hour>=12 and hour<18:
        t="Good afternoon"

    else:
        t="Good evening" 
    greetinglist=["How may I help you?","How can I be of service today?","Good to see you again.","Hope you are well?"]
    speak(t+", Sir. "+random.choice(greetinglist))       


def takeCommand():
    #It takes microphone input from the user and returns string output

    r = sr.Recognizer()
    with sr.Microphone() as source:
        speak("Will be listening after window closes...")
        r.pause_threshold = 1
        audio = r.listen(source)
    
    try:
        speak("Recognizing...")    
        query = r.recognize_google(audio, language='en-us')
        
    except Exception as e:
        print(e)
        errorlist=["Sir, can you please say that again please?"," I haven\'t quite caught that.","Excuse me?"]
        speak(random.choice(errorlist))  
        return "None"
    if query.startswith('hey Jarvis'):    
        return query
    elif 'exit' in query or 'quit' in query:
        sys.exit()
    else:
        return "None"

    
def sendEmail(to, content):
    server = smtplib.SMTP('smtp.qq.com', 587)
    server.ehlo()
    server.starttls()
    server.login('joelzc2023@qq.com', 'wmaKX2018')
    server.sendmail('joelzc2023@qq.com', to, content)
    server.close()


def net(url):
    try:
        webbrowser.open_new_tab(url)
    except:
        webbrowser.open(url)
        
        
        
if __name__ == "__main__":
    wishMe()
    while True:
    # if 1:
        query = takeCommand().lower()
        logic=''
        # Logic for executing tasks based on query
        # TODO: add more logic
        if 'wikipedia' in query or query.startswith('what is'):
            logic='wikipedia'
        if 'open youtube' in query:
            logic='youtube'
        if 'open google' in query:
            logic='google'
        if 'open stackoverflow' in query:
            logic='stackoverflow'
        
        if logic=='wikipedia':
            speak('Searching Wikipedia...')
            try:
                query = query.replace("wikipedia", "")
                query = query.replace("hey Jarvis","")
                query = query.replace("what is","")
            except:
                pass
            
            results = wikipedia.summary(query, sentences=2)
            speak("According to Wikipedia\n"+results)
            

        elif logic=='youtube':
            
            net("https://www.youtube.com")

        elif logic=='google':
            net("https://www.google.com")

        elif logic=='stackoverflow':
            net("https://www.stackoverflow.com")   


        elif 'play music' in query:
            music_dir = r'D:\Music'
            songs = os.listdir(music_dir)
            window=sg.Window('J.A.R.V.I.S.',music_layout)
            while True:
                    event, values = window.read()
                    if event=='Shuffle music':
                        os.startfile(os.path.join(music_dir, random.choice(songs)))
                        break
                    if event == sg.WIN_CLOSED or event == 'Cancel': 
                        break 
            window.close()
                

        elif 'the time' in query:
            strTime = datetime.datetime.now().strftime("%H:%M:%S")    
            speak(f"Sir, the time is {strTime}")
        
        

        # elif 'open code' in query:
        #     codePath = "C:\\Users\\Haris\\AppData\\Local\\Programs\\Microsoft VS Code\\Code.exe"
        #     os.startfile(codePath)

        elif 'email ' in query:
            try:
                # Create the Window
                window = sg.Window('J.A.R.V.I.S.', email_layout)
                # Event Loop to process "events" and get the "values" of the inputs
                while True:
                    event, values = window.read()
                    if event=='Ok':
                        content=values[0]
                        to=values[1]
                        break
                    if event == sg.WIN_CLOSED or event == 'Cancel': # if user closes window or clicks cancel
                        break 
                window.close()

                if to=='Shannon':
                    to='shannon.gcy1107@gmail.com'
                elif to=='Leah':
                    to='leah8kw@gmail.com'
                elif to=='Caelyn':
                    to='caelynwang23@163.com'
                sendEmail(to, content)
                speak("Email has been sent!")
            except Exception as e:
                print(e)
                speak("ERROR")    
