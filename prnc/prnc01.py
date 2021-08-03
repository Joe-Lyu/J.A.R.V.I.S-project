# Author:Tom
# test the pronunciation of project
# read the line of the all.txt line by line

# * pip install pyttsx3 *

import pyttsx3

f = open("all.txt", 'r')
line = f.readline()
engine = pyttsx3.init('sapi5', True)

# rate = engine.getProperty('rate')   # getting details of current speaking rate
# print (rate)                        # printing current voice rate
engine.setProperty('rate', 250)  # setting up new voice rate
voices = engine.getProperty('voices')  # get system voice pack
engine.setProperty('voice', voices[2].id)  # changing index, changes voices. 2 for male(FIXME:change for different user
#engine.setProperty('volume', 1.0)  # volume level (between 0 and 1)

while line:
    line = f.readline()
    print(line, end='')
    engine.say(line)
    engine.runAndWait()
    #engine.say("Hello World!")
f.close()
