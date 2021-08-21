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
# JARVIS V1.3.0 BETA
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
# FIXED(1.3.0)      slow recognition time;     (->use sphinx)
#                   limited abilities & responses;
# FIXING            limited understanding;    (->enlarge database)
#                   no conversational skills;
# FIXED(1.0.1)      proxy problems;
# FIXED(1.0.2)      web browser problems
# FIXED(1.2.2)      improving understanding   (->reduce intents)
# =============================================================================
# =============================================================================
# fixed proxys;
# minor change to 'quit program';
# minor change to wikipedia function;
# added silent mode for those who cant access google or dont want to be heard;
# added quit notification;
# added search in google
# switched voice recognition engine to sphinx for offline usage
# =============================================================================
import os

import init_venv
from reply_func import speak, wiki, wishMe, sendEmail, net
from utils import SilentlyTakeCommand, takeCommand

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # FIXME: mandatory use cpu to train
while 1:
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
        import PySimpleGUI as pySG
        import numpy as np
        import pandas as pd
        import pickle
        from silence_tensorflow import silence_tensorflow
        # silence_tensorflow()
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
        break
    except ModuleNotFoundError as e:
        # TODO: swig can't be download by pip, it needs to be downloaded in http://www.swig.org/download.html,
        # and use requests and set env variable of path
        # remove pickle, because it built in python
        res = init_venv.install_package_check(
            ["PyAudio", "PySimpleGUI", "SpeechRecognition", "matplotlib", "nltk", "numpy", "pandas", "pyttsx3",
             "pywin32",
             "scikit_learn", "silence_tensorflow", "tensorflow", "wikipedia", "re", "pocketsphinx"])
        if not res[1]:
            print("These packages cannot be installed successfully: ", res)
            os.close(-1)
        break

# proxy
proxy = "http://127.0.0.1:11223"  # FIXME:need to change
os.environ['http_proxy'] = proxy
os.environ['HTTP_PROXY'] = proxy
os.environ['https_proxy'] = proxy
os.environ['HTTPS_PROXY'] = proxy

# load dataset
df = pd.read_csv(sys.path[0] + "/datasets/Dataset-train.csv", encoding="latin1", names=["Sentence", "Intent"])
intent = df["Intent"]
unique_intent = list(set(intent))
sentences = list(df["Sentence"])


def cleaning(inp: list) -> list:
    """
    sentence List -> tokenized sentence List
    :param inp: sentence
    :return: tokenized sentence List
    """
    words = []
    for s in inp:
        # substitute none alphabet and num into space (sentences by sentences)
        clean = re.sub(r'[^a-zA-Z0-9]', " ", s)
        w = word_tokenize(clean)
        # stemming
        words.append([i.lower() for i in w])
    return words


def create_tokenizer(words, filters='!"#$%&()*+,-./<=>:;?@[]^_`{|}~\\'):
    token = Tokenizer(filters=filters)
    token.fit_on_texts(words)
    return token


def encoding_doc(token, words):  # 编码
    return token.texts_to_sequences(words)


def padding_doc(enc_doc, max_len):
    return pad_sequences(enc_doc, maxlen=max_len, padding="post")


# for sentence
cleaned_words = cleaning(sentences)
word_tokenizer = create_tokenizer(cleaned_words)
encoded_doc = encoding_doc(word_tokenizer, cleaned_words)
# vocab_size = len(word_tokenizer.word_index) + 1 #didn't used?
max_length = len(max(cleaned_words, key=len))  # longest sentence word-count
padded_doc = padding_doc(encoded_doc, max_length)

# for intent
output_tokenizer = create_tokenizer(unique_intent, filters='!"#$%&()*+,-/:;<=>?@[]^`{|}~\\')
encoded_output = encoding_doc(output_tokenizer, intent)
x = []
for i in encoded_output:
    x.append(i[0])
encoded_output = x
encoded_output = (np.array(encoded_output).reshape(len(encoded_output), 1))
output_one_hot = OneHotEncoder(sparse=False).fit_transform(encoded_output)
train_X, val_X, train_Y, val_Y = train_test_split(padded_doc, output_one_hot, test_size=0.2)

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
checkpoint = ModelCheckpoint('intent.h5', verbose=1, save_best_only=True, mode='min')
model = load_model("intent.h5")


# hist= model.fit(train_X,train_Y,epochs=105,batch_size=32,validation_data=(val_X,val_Y),callbacks=[checkpoint])


def predictions(text):
    clean = re.sub(r'[^ a-zA-Z0-9]', " ", text)
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


pySG.theme('DarkAmber')  # Iron Man theme (as close as I can get)
# layout of all the windows
email_layout = [[pySG.Text('J.A.R.V.I.S. email service')],
                [pySG.Text('Email content'), pySG.InputText()],
                [pySG.Text('send to'), pySG.InputText()],
                [pySG.Button('Ok'), pySG.Button('Cancel')]]
music_layout = [[pySG.Text('J.A.R.V.I.S. music player')],
                [pySG.Button('Shuffle music'), pySG.Button('Cancel')]]

# browser settings
# env variable
chrome_path = "chrome.exe"  # true on windows if not reinstalled

webbrowser.register('chrome', webbrowser.BackgroundBrowser(chrome_path), 1)
webbrowser.get('chrome')


def main(command):
    intent, confidence = get_final_output(predictions(command), unique_intent)
    print(intent) if confidence >= 0.7 else print('unknown', intent, confidence)
    logic = ''
    if command == "UNKNOWN_COMMAND":
        error_list = ["Sir, can you please say that again please?", " I haven\'t quite caught that.", "Excuse me?"]
        speak(random.choice(error_list))
        # Logic for executing tasks based on query
    # TODO: add more logic
    if 'wikipedia' in command or command.startswith('what is'):
        logic = 'wikipedia'
    elif 'youtube' in command:
        logic = 'youtube'
    elif 'search' in command and 'in google' in command:
        logic = 'search google'
        command = command.replace("search for", '')
        command = command.replace('in google', '')
        command = command.replace(" ", '+')
    elif 'google' in command:
        logic = 'google'
    elif 'stackoverflow' in command:
        logic = 'stackoverflow'
    else:
        error_list = ["Sir, can you please say that again please?", " I haven\'t quite caught that.", "Excuse me?"]
        speak(random.choice(error_list))
    if logic == 'wikipedia':
        command = command.replace("search wikipedia for", "")
        command = command.replace("hey jarvis", "")
        command = command.replace("what is", "")
        speak('Searching Wikipedia for' + command + '...')
        results = wikipedia.summary(command, sentences=2)
        wiki("According to Wikipedia,\n" + results)
    elif logic == 'youtube':
        net("https://www.youtube.com")
    elif logic == 'google':
        net("https://www.google.com")
    elif logic == 'stackoverflow':
        net("https://www.stackoverflow.com")
    elif logic == 'search google':
        net("https://www.google.com/search?q=" + command)
    elif 'play music' in command:
        music_dir = r'D:\Music'
        songs = os.listdir(music_dir)
        window = pySG.Window('J.A.R.V.I.S.', music_layout)
        while True:
            event, values = window.read()
            if event == 'Shuffle music':
                os.startfile(os.path.join(music_dir, random.choice(songs)))
                break
            if event == pySG.WIN_CLOSED or event == 'Cancel':
                break
        window.close()
    elif 'the time' in command:
        strTime = datetime.datetime.now().strftime("%H:%M:%S")
        speak(f"Sir, the time is {strTime}")
    elif 'email ' in command:
        try:
            # Create the Window
            window = pySG.Window('J.A.R.V.I.S.', email_layout)
            content, to = "", ""
            while True:
                event, values = window.read()
                if event == 'Ok':
                    content = values[0]
                    to = values[1]
                    break
                if event == pySG.WIN_CLOSED or event == 'Cancel':
                    break
            window.close()
            sendEmail(to, content)
            speak("Email has been sent!")
        except Exception as e:
            print(e)
            speak("ERROR")


if __name__ == "__main__":
    wishMe()
    SILENT_MODE, VOICE_MODE = 0, 1
    mode = SILENT_MODE
    while True:
        if mode == SILENT_MODE:
            query = SilentlyTakeCommand().lower()
        else:
            query = takeCommand().lower()
        if query == 'voice mode':
            mode = VOICE_MODE
        elif query == 'silent mode':
            mode = SILENT_MODE
        main(query)
