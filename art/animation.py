# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 00:18:37 2021

@author: DELL
"""
import time

from PIL import Image, ImageTk, ImageSequence
import PySimpleGUI as sg
import sys
import random
gif_filename = sys.path[0] + r'\\resized.gif'
sg.theme('DarkAmber')  # Iron Man theme (as close as I can get)

layout = [[sg.Image(key='-IMAGE-')]]

window = sg.Window('J.A.R.V.I.S.', layout, element_justification='c', margins=(0, 0), element_padding=(0, 0),
                   finalize=True)

sequence = [ImageTk.PhotoImage(img) for img in
            ImageSequence.Iterator(Image.open(gif_filename))]  # must has finalized to do this

interframe_duration = Image.open(gif_filename).info['duration']  # get how long to delay between frames

frame = random.choice(sequence)
while True:
    direction = random.choice([-1, 1, 1])
    length = random.randint(1, 45)
    ind = sequence.index(frame)
    if ind <= length and direction == -1:
        direction = 1
    elif ind > length and direction == 1:
        direction = -1
    newseq = []
    for i in range(1, length + 1):
        newseq.append(sequence[ind + i * direction])
    for i in newseq:
        event, values = window.read(timeout=interframe_duration)
        if event == sg.WIN_CLOSED:
            sys.exit()
        window['-IMAGE-'].update(data=i)
    frame = newseq[-1]
    time.sleep(interframe_duration / 1000)
