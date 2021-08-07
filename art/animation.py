# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 00:18:37 2021

@author: DELL
"""

from PIL import Image, ImageTk, ImageSequence
import PySimpleGUI as sg
import sys

gif_filename = sys.path[0] + r'\\resized.gif'
sg.theme('DarkAmber')  # Iron Man theme (as close as I can get)


layout = [[sg.Image(key='-IMAGE-')]]

window = sg.Window('J.A.R.V.I.S.', layout, element_justification='c', margins=(0,0), element_padding=(0,0), finalize=True)

sequence = [ImageTk.PhotoImage(img) for img in ImageSequence.Iterator(Image.open(gif_filename))]    # must has finalized to do this

interframe_duration = Image.open(gif_filename).info['duration']     # get how long to delay between frames

while True:
    for frame in sequence:
        event, values = window.read(timeout=interframe_duration)
        if event == sg.WIN_CLOSED:
            sys.exit()
        window['-IMAGE-'].update(data=frame)
