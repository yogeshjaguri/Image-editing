from tkinter import *
from PIL import ImageTk
import PIL.Image

integrity = Tk()

integrity.geometry("500x500")
#photo = PhotoImage(file ="1.png")  # not for jpg

# for jpg Image
image = PIL.Image.open("1.jpg")
photo = ImageTk.PhotoImage(image)


Label1 = Label(image=photo)
Label1.pack()


integrity.mainloop()
