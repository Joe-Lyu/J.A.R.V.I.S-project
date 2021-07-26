# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 23:37:34 2021

@author: DELL
"""
# =============================================================================
# JARVIS V1.1.0 BETA
# features: 
#                   voice control and type control;
#                   open several websites;
#                   play music (suffle only);
#                   tell time;
#                   send email
# issues:
#                   slow recognition time;
#                   limited abilities & responses;
#                   limited understanding;
#                   no conversational skills;
#FIXED(1.0.1)       proxy problems;
#FIXED(1.0.2)       web browser problems
# =============================================================================
# =============================================================================
# fixed proxys;
# minor change to 'quit program';
# minor change to wikipedia function
# added silent mode for those who cant access google or dont want to be heard
# added quit notification
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



sg.theme('DarkAmber')   #Iron Man theme (as close as I can get)

#layout of all the windows
email_layout = [  [sg.Text('J.A.R.V.I.S. email service')],
            [sg.Text('Email content'), sg.InputText()],
            [sg.Text('send to'),sg.InputText()],
            [sg.Button('Ok'), sg.Button('Cancel')] ]
music_layout = [  [sg.Text('J.A.R.V.I.S. music player')],
                [sg.Button('Shuffle music'),sg.Button('Cancel')]]


#browser settings
chrome_path="C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe"

webbrowser.register('chrome', webbrowser.BackgroundBrowser(chrome_path),1)
webbrowser.get('chrome')


def speak(msg):
    speak_layout = [ [sg.Text(msg)]]
    window=sg.Window('J.A.R.V.I.S.', speak_layout,auto_close=True,auto_close_duration=2)
    event, values = window.read()

    
def wiki(content):
    wiki_layout=[ [sg.Text(content)],[sg.Button('Ok')]]
    window=sg.Window('J.A.R.V.I.S.', wiki_layout)
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == 'Ok': 
            break 
    window.close()


def wishMe():
    hour = int(datetime.datetime.now().hour)
    if hour>=4 and hour<12:
        t="Good morning"

    elif hour>=12 and hour<18:
        t="Good afternoon"

    else:
        t="Good evening" 
    greetinglist=["How may I help you?","How can I be of service today?","Good to see you again.","Hope you are well?"]
    speak(t+", sir. "+random.choice(greetinglist))       

def SilentlyTakeCommand():
    type_layout=[[sg.Text('Type your command, sir.')],
             [sg.InputText()],
             [sg.Button("Confirm",bind_return_key=True)]]
    window=sg.Window('J.A.R.V.I.S.',type_layout)
    while True:
        event, values = window.read()
        if event=='Confirm':
            query=values[0]
            break
    window.close()
    if 'exit' in query or 'quit' in query:
        speak("Quitting program...\nGoodbye, sir.")
        sys.exit()
    return query



def takeCommand():
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
    #print(query)                             #this line is for debugging
    if query.startswith('hey Jarvis'):    
        return query
    elif 'exit' in query or 'quit' in query:
        speak("Quitting program...\nGoodbye, sir.")
        sys.exit()
    elif query=='silent mode' or query=='silence mode':
        return query
    else:
        return "None"

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
        
def main(query):
    if query=="None":
        errorlist=["Sir, can you please say that again please?"," I haven\'t quite caught that.","Excuse me?"]
        speak(random.choice(errorlist))  
    logic=''
    # Logic for executing tasks based on query
    # TODO: add more logic
    if 'search wikipedia for' in query or query.startswith('what is'):
        logic='wikipedia'
    if 'youtube' in query:
        logic='youtube'
    if 'google' in query:
        logic='google'
    if 'stackoverflow' in query:
        logic='stackoverflow'
    
        
        
        
    if logic=='wikipedia':
        query = query.replace("search wikipedia for", "")
        query = query.replace("hey jarvis","")
        query = query.replace("what is","")

        speak('Searching Wikipedia for'+query+'...')
        
        results = wikipedia.summary(query, sentences=2)
        wiki("According to Wikipedia,\n"+results)
        
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

    elif 'email ' in query:
        try:
            # Create the Window
            window = sg.Window('J.A.R.V.I.S.', email_layout)
            while True:
                event, values = window.read()
                if event=='Ok':
                    content=values[0]
                    to=values[1]
                    break
                if event == sg.WIN_CLOSED or event == 'Cancel': 
                    break 
            window.close()

            speak("Email has been sent!")
        except Exception as e:
            print(e)
            speak("ERROR")    
        
        
        
        
        
if __name__ == "__main__":
    wishMe()
    while True:
        while True:
            query = SilentlyTakeCommand().lower()
            if query=='voice mode':
                break
            else:
                main(query)
        while True:
            query = takeCommand().lower()
            if query=='silent mode':
                break
            else:
                main(query)
        
