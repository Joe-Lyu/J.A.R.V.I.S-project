import random
import sys

import PySimpleGUI as sg
import speech_recognition as sr

from reply_func import speak


def SilentlyTakeCommand():
    type_layout = [[sg.Text('Type your command, sir.')],
                   [sg.InputText()],
                   [sg.Button("Confirm", bind_return_key=True)]]
    window = sg.Window('J.A.R.V.I.S.', type_layout)
    while True:
        event, values = window.read()
        if event == 'Confirm':
            query = values[0]
            break
    window.close()
    if 'exit' in query or 'quit' in query:
        speak("Quitting program...\nGoodbye.", "Quitting program...\nGoodbye, sir.")
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
        errorlist = ["Sir, can you please say that again please?", " I haven\'t quite caught that.", "Excuse me?"]
        speak(random.choice(errorlist))
        return "None"
    # print(query)                             #this line is for debugging
    if query.startswith('hey Jarvis'):
        return query
    elif 'exit' in query or 'quit' in query:
        speak("Quitting program...\nGoodbye.")
        sys.exit()
    elif query == 'silent mode' or query == 'silence mode':
        query = 'silent mode'
        return query
    else:
        return "UNKNOWN_COMMAND"