# testing on adding gif
import PySimpleGUI as sg
from PIL import Image
import sys

gif_filename = sys.path[0] + r'\\resized.gif'

gif_file = Image.open(gif_filename)
size = gif_file.size
del gif_file
sg.theme('DarkAmber')
img = sg.Image(gif_filename, key="-GIF-", size=size, background_color="black")
layout = [
    [sg.Quit()],
    [img]
]
window = sg.Window('J.A.R.V.I.S.', layout, element_justification='c', margins=(0, 0), element_padding=(0, 0),
                   finalize=True)
# img = sg.Image(gif_filename)
while True:
    event, values = window.read(timeout=10, timeout_key="-TIMEOUT-")

    # way to quit
    if event in (sg.WIN_CLOSED, "Quit", "Cancel"):
        break

    img.update_animation(gif_filename, time_between_frames=50)
window.close()
