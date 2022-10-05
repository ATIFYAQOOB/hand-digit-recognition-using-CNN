import tkinter as tk
from tkinter import *
from tkinter import messagebox
from PIL import Image,ImageTk

from keras.models import load_model
import cv2
import numpy as np
import pyscreenshot as ImageGrab
window = tk.Tk()
window.title("Hand Digit Recognition App")
window.resizable(0,0)

canvas1 = Canvas(window , width= 400 , height = 400 , bg = "black" )
canvas1.place(x=10 , y= 50)

canvas2 = Canvas(window , width= 400 , height = 400 , bg = "pink")
canvas2.place(x=340, y= 50)

def activate_paint(e):
    global lastx , lasty
    canvas1.bind("<B1-Motion>" , paint)
    lastx , lasty = e.x , e.y
def paint(e):
   
    x,y = e.x , e.y
    global lastx , lasty
    canvas1.create_line(lastx  , lasty ,x ,y , fill = "white", width = 25 ,capstyle= ROUND, smooth = True  )
canvas1.bind('<1>', activate_paint)

def clear():
    canvas1.delete("all")
btn = tk.Button(window , text = "CLEAR" , fg = "white" ,bg = "green" , command = clear)
btn.place(x = 4 , y=5)

def predict():
    
    model = load_model("model.h5")
    x = canvas1.winfo_rootx()
    y = canvas1.winfo_rooty()
    x1 = x + canvas1.winfo_width()
    y1 = y + canvas1.winfo_height()
    img = ImageGrab.grab(bbox = (x , y ,x1, y1))
    img.save("paint.png")
    img = cv2.imread("paint.png")
    load = Image.open("paint.png")
    load = load.resize((300,400))
    photo = ImageTk.PhotoImage(load)
    img = Label(canvas2, image = photo , width = 300 , height = 400)
    img.image = photo
    img.place(x = 0 , y=0)
    img = cv2.imread("paint.png",cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img,(28,28))
    data = np.array(img)
    img_data = float(data)
    img_data = img_data/255.0
    data = img_data.reshape(-1,28,28,1)
    output = model.predict([data])
    a = tk.Label(canvas2 , text = "Prediction:" + str(np.argmax(output)) , font = ("Algerian", 20))
    a.place(x = 0, y = 100)

bt = tk.Button(window , text = "Predict" , fg = 'white' , bg = "blue" ,command = predict)
bt.place(x = 50 , y = 5)

window.geometry("650x500")
window.mainloop()
