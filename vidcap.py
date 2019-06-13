import cv2
import PIL
import os
import tkinter as tk
import tkinter as tk

cap = cv2.VideoCapture(0)


def cap_img():
    ret, frame = cap.read()
    gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("your hand", gray_scale)


root = tk.Tk()
frame = tk.Frame(root)
frame.pack()

button = tk.Button(frame,text="QUIT", fg="red", command=quit)
button.pack(side=tk.LEFT)
slogan = tk.Button(frame,
                   text="Click the button to take a picture!!!",
                   command=cap_img())
slogan.pack(side=tk.LEFT)

root.mainloop()

cap.release()
cv2.destroyAllWindows()
