
import webbrowser
import os
import threading

def launchTensorBoard():
    os.system('tensorboard --logdir=logs/')
    return

def run():
    tb = threading.Thread(target=launchTensorBoard, args=([]))
    tb.start()
    webbrowser.open('http://localhost:6006')