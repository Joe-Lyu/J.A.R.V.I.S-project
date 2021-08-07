# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 23:37:34 2021
pip install:================================================================= FIXME:Can this be done automatically
Pillow	8.3.1	8.3.1
PyAudio	0.2.11	0.2.11 (might need whl)
PySimpleGUI	4.45.0	4.45.0
SpeechRecognition	3.8.1	3.8.1
matplotlib	3.4.2
nltk	3.6.2
numpy	1.19.5
pandas	1.3.1
pyttsx3	2.90
pywin32	301
scikit-learn	0.24.2
silence-tensorflow	1.1.1
tensorflow	2.5.0
wikipedia	1.4.0
=============================================================================
please: set the path and check other FIXMEs
        unzip intent.zip TODO: done by program
@author: Joe Tom
"""
# =============================================================================
# JARVIS V1.2.2 BETA
# features:
#                   autopiptool;(v1.2.2)
#                   speak;(v1.2.1)
#                   voice control and type control;(v1.0.2)
#                   open several websites;(v1.0.2)
#                   play music (suffle only);(v1.0.1)
#                   tell time;(v1.0.1)
#                   send email;(v1.0.1)
# issues:
#                   awkward pronunciation;     (->search better lib)
#                   slow recognition time;
#                   limited abilities & responses;
# FIXING             limited understanding;    (->enlarge database)
#                   no conversational skills;
# FIXED(1.0.1)       proxy problems;
# FIXED(1.0.2)       web browser problems
# FIXED(1.2.2)       improving understanding   (->reduce intents)
# =============================================================================
# =============================================================================
# fixed proxys;
# minor change to 'quit program';
# minor change to wikipedia function;
# added silent mode for those who cant access google or dont want to be heard;
# added quit notification;
# added search in google
# =============================================================================
from init_venv import *
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"  #FIXME manditorily use cpu to train
pipList = ["PyAudio", "PySimpleGUI", "SpeechRecognition", "matplotlib", "nltk", "numpy", "pandas", "pyttsx3", "pywin32",
           "scikit_learn", "silence_tensorflow", "tensorflow", "wikipedia", "re"]

# import...
try:
    import pyttsx3
    import speech_recognition as sr
    import datetime
    import wikipedia
    import webbrowser
    import os
    import sys
    import smtplib
    import random
    import PySimpleGUI as sg
    import numpy as np
    import pandas as pd
    from silence_tensorflow import silence_tensorflow

    #silence_tensorflow()
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem.lancaster import LancasterStemmer
    import nltk
    import re
    from sklearn.preprocessing import OneHotEncoder
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.callbacks import ModelCheckpoint
    from sklearn.model_selection import train_test_split
    from tensorflow.keras import layers
    import tensorflow as tf
    from tensorflow.keras.models import load_model
except ModuleNotFoundError as e:
    check = install_package_check(pipList)
    if check is not True:
        print("FalseList=", check)
finally:
    import pyttsx3
    import speech_recognition as sr
    import datetime
    import wikipedia
    import webbrowser
    import os
    import sys
    import smtplib
    import random
    import PySimpleGUI as sg
    import numpy as np
    import pandas as pd
    from silence_tensorflow import silence_tensorflow

    silence_tensorflow()
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem.lancaster import LancasterStemmer
    import nltk
    import re
    from sklearn.preprocessing import OneHotEncoder
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.callbacks import ModelCheckpoint
    from sklearn.model_selection import train_test_split
    from tensorflow.keras import layers
    import tensorflow as tf
    from tensorflow.keras.models import load_model

# proxy
proxy = "http://127.0.0.1:15732"  # FIXME:need to change
os.environ['http_proxy'] = proxy
os.environ['HTTP_PROXY'] = proxy
os.environ['https_proxy'] = proxy
os.environ['HTTPS_PROXY'] = proxy

# voice initiation
engine = pyttsx3.init('sapi5', True)
engine.setProperty('rate', 225)  # setting up new voice rate
voices = engine.getProperty('voices')  # get system voice pack
engine.setProperty('voice', voices[2].id)  # changing index, changes voices. 2 for male

PROJECT_PATH = sys.path[0]
print("PATH=",sys.path[0])

def load_dataset(filename):
    df = pd.read_csv(filename, encoding="latin1", names=["Sentence", "Intent"])
    intent = df["Intent"]
    unique_intent = list(set(intent))
    sentences = list(df["Sentence"])

    return intent, unique_intent, sentences


intent, unique_intent, sentences = load_dataset(PROJECT_PATH + r"\Dataset-train.csv")
#stemmer = LancasterStemmer()


def cleaning(sentences):  # sentence List->tokenized sentence List
    words = []
    for s in sentences:
        clean = re.sub(r'[^ a-z A-Z 0-9]', " ", s)  #substitute none alphabet and num into space (sentences by sentences)
        w = word_tokenize(clean)
        # stemming
        words.append([i.lower() for i in w])

    return words


def create_tokenizer(words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'):
    token = Tokenizer(filters=filters)
    token.fit_on_texts(words)
    return token


def get_max_length(words):  # longest sentence word-count
    return len(max(words, key=len))


def encoding_doc(token, words):  # 编码
    return token.texts_to_sequences(words)


def padding_doc(encoded_doc, max_length):
    return pad_sequences(encoded_doc, maxlen=max_length, padding="post")

#for sentence
cleaned_words = cleaning(sentences)
word_tokenizer = create_tokenizer(cleaned_words)
encoded_doc = encoding_doc(word_tokenizer, cleaned_words)
# vocab_size = len(word_tokenizer.word_index) + 1 #didn't used?
max_length = get_max_length(cleaned_words)
padded_doc = padding_doc(encoded_doc, max_length)

# for intent
output_tokenizer = create_tokenizer(unique_intent, filters='!"#$%&()*+,-/:;<=>?@[\]^`{|}~')
encoded_output = encoding_doc(output_tokenizer, intent)

x = []
for i in encoded_output:
    x.append(i[0])
encoded_output = x

encoded_output = (np.array(encoded_output).reshape(len(encoded_output), 1))


def one_hot(encode):
    o = OneHotEncoder(sparse=False)
    return o.fit_transform(encode)


output_one_hot = one_hot(encoded_output)

train_X, val_X, train_Y, val_Y = train_test_split(padded_doc, output_one_hot, shuffle=True, test_size=0.2)

max_features = 15000
embedding_dim = 128
sequence_length = 500
inputs = tf.keras.Input(shape=(None,), dtype="int64")

x = layers.Embedding(max_features, embedding_dim)(inputs)
x = layers.Dropout(0.5)(x)

x = layers.Conv1D(128, 6, padding="same", activation="relu", strides=3)(x)
x = layers.Conv1D(128, 6, padding="same", activation="relu", strides=3)(x)
x = layers.GlobalMaxPooling1D()(x)

x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.5)(x)

predictions = layers.Dense(39, activation="sigmoid", name="predictions")(x)

model = tf.keras.Model(inputs, predictions)

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

filename = 'intent.h5'
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
model = load_model("intent.h5")
#hist= model.fit(train_X,train_Y,epochs=105,batch_size=32,validation_data=(val_X,val_Y),callbacks=[checkpoint])


def predictions(text):
    clean = re.sub(r'[^ a-z A-Z 0-9]', " ", text)
    test_word = word_tokenize(clean)
    test_word = [w.lower() for w in test_word]
    test_ls = word_tokenizer.texts_to_sequences(test_word)
    # Check for unknown words
    if [] in test_ls:
        test_ls = list(filter(None, test_ls))

    test_ls = np.array(test_ls).reshape(1, len(test_ls))

    x = padding_doc(test_ls, max_length)

    pred = model.predict(x)

    return pred


def get_final_output(pred, classes):
    predictions = pred[0]

    classes = np.array(classes)
    ids = np.argsort(-predictions)
    classes = classes[ids]
    predictions = -np.sort(-predictions)

    return classes[0], predictions[0]


sg.theme('DarkAmber')  # Iron Man theme (as close as I can get)

# layout of all the windows
email_layout = [[sg.Text('J.A.R.V.I.S. email service')],
                [sg.Text('Email content'), sg.InputText()],
                [sg.Text('send to'), sg.InputText()],
                [sg.Button('Ok'), sg.Button('Cancel')]]
music_layout = [[sg.Text('J.A.R.V.I.S. music player')],
                [sg.Button('Shuffle music'), sg.Button('Cancel')]]

# browser settings
chrome_path = "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe" #true on windows if not reinstalled

webbrowser.register('chrome', webbrowser.BackgroundBrowser(chrome_path), 1)
webbrowser.get('chrome')


def speak(msg, msg_show=0):  # popup windows actually
    engine.say(msg)
    if (msg_show != 0):
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
    pred = predictions(query)
    intent, confidence = get_final_output(pred, unique_intent)
    print(intent) if confidence >= 0.7 else print('unknown', intent, confidence)
    logic = ''
    if query == "UNKNOWN_COMMAND":
        errorlist = ["Sir, can you please say that again please?", " I haven\'t quite caught that.", "Excuse me?"]
        speak(random.choice(errorlist))
        # Logic for executing tasks based on query
    # TODO: add more logic
    if 'wikipedia' in query or query.startswith('what is'):
        logic = 'wikipedia'
    elif 'youtube' in query:
        logic = 'youtube'
    elif 'search' in query and 'in google' in query:
        logic = 'search google'
        query = query.replace("search for", '')
        query = query.replace('in google', '')
        query = query.replace(" ", '+')
    elif 'google' in query:
        logic = 'google'
    elif 'stackoverflow' in query:
        logic = 'stackoverflow'
    else:
        errorlist = ["Sir, can you please say that again please?", " I haven\'t quite caught that.", "Excuse me?"]
        speak(random.choice(errorlist))

    if logic == 'wikipedia':
        query = query.replace("search wikipedia for", "")
        query = query.replace("hey jarvis", "")
        query = query.replace("what is", "")

        speak('Searching Wikipedia for' + query + '...')

        results = wikipedia.summary(query, sentences=2)
        wiki("According to Wikipedia,\n" + results)

    elif logic == 'youtube':

        net("https://www.youtube.com")

    elif logic == 'google':
        net("https://www.google.com")

    elif logic == 'stackoverflow':
        net("https://www.stackoverflow.com")

    elif logic == 'search google':
        net("https://www.google.com/search?q=" + query)

    elif 'play music' in query:
        music_dir = r'D:\Music'
        songs = os.listdir(music_dir)
        window = sg.Window('J.A.R.V.I.S.', music_layout)
        while True:
            event, values = window.read()
            if event == 'Shuffle music':
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
                if event == 'Ok':
                    content = values[0]
                    to = values[1]
                    break
                if event == sg.WIN_CLOSED or event == 'Cancel':
                    break
            window.close()
            sendEmail(to, content)
            speak("Email has been sent!")
        except Exception as e:
            print(e)
            speak("ERROR")


if __name__ == "__main__":
    wishMe()
    while True:
        while True:
            query = SilentlyTakeCommand().lower()
            if query == 'voice mode':
                break
            else:
                main(query)
        while True:
            query = takeCommand().lower()
            if query == 'silent mode':
                break
            else:
                main(query)
