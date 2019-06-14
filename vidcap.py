import cv2
import PIL
import os
import numpy as np
import tkinter as tk

cap = cv2.VideoCapture(0)


def cap_img():
    ret, frame = cap.read()
    gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_img =  cv2.resize(gray_scale,(28,28))
    cv2.imwrite('new_img.jpg',resized_img )

root = tk.Tk()
frame = tk.Frame(root)
frame.pack()

button = tk.Button(frame, text="QUIT", fg="red", command=quit)
button.pack(side=tk.LEFT)
title = tk.Button(frame, text="Click the button to take a picture!!!", command=cap_img())
title.pack(side=tk.LEFT)

root.mainloop()
cap.release()
cv2.destroyAllWindows()
