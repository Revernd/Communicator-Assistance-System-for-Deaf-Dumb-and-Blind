import cv2
from tkinter import *
import tkinter.messagebox
from Speech_To_Text import run_speech 
from Text_To_Speech import run_text
from SignDetect import sign_Detection
from SignSlides import sign_slides
from collections import OrderedDict
from textblob import TextBlob
from spellchecker import SpellChecker


root=Tk()
root.geometry('500x570')
root.title('Communicator Assistance Network App')

bg = PhotoImage(file="Dataset/sign-deaf-disabled.png", master=root)

my_canvas = Canvas(root, width=500, height=570)
my_canvas.pack(fill="both", expand=True)

# Set image in canvas
my_canvas.create_image(0,0, image=bg, anchor="nw")


def hel():
   help(cv2)

def Contri():
   tkinter.messagebox.showinfo("Contributors","\n1.A.T.Ruthvik Srinivas Deekshitulu\n2.C.H.Adityavardhan \n3. Prof.Hemprasad Patil \n3. Prof.Surya Prakash\n")


def anotherWin():
   tkinter.messagebox.showinfo("About",'Communicator Assistance Network version v1.0\n Made Using\n-OpenCV\n-Keras\n-Tkinter\n In Python 3')
                                    
   

menu = Menu(root)
root.config(menu=menu)

subm1 = Menu(menu)
menu.add_cascade(label="Tools",menu=subm1)
subm1.add_command(label="Open CV Docs",command=hel)

subm2 = Menu(menu)
menu.add_cascade(label="About",menu=subm2)
subm2.add_command(label="CAN Configuration",command=anotherWin)
subm2.add_command(label="Contributors",command=Contri)

def Speech_to_text():
    text = run_speech()
    sign_slides(text)
 
def all_gestures():
    images = cv2.imread("Dataset/all_signs.jpg")
    cv2.imshow("All Signs", images)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def text_formation():
    global text_final
    text = sign_Detection()
    text_rem_dup = "".join(OrderedDict.fromkeys(text))
    text_no_digit = ''.join((x for x in text_rem_dup if not x.isdigit()))
    text_blob = TextBlob(text_no_digit)
    # spell = SpellChecker()
    # text_spell = spell.unknown(text_blob.correct().lower())
    # text_pred = spell.correction(text_spell)
    text_final = text_blob.correct().lower()

   
but2=Button(root,padx=5,pady=5,width=39,bg='white',fg='black',relief=GROOVE,command=text_formation,text='Sign Detection',font=('helvetica 15 bold'))
but2.place(x=5,y=176)

but3=Button(root,padx=5,pady=5,width=39,bg='white',fg='black',relief=GROOVE,command=lambda : run_text(text_final),text='Text_to_Speech',font=('helvetica 15 bold'))
but3.place(x=5,y=250)

but4=Button(root,padx=5,pady=5,width=39,bg='white',fg='black',relief=GROOVE,command=Speech_to_text,text='Speech_to_text',font=('helvetica 15 bold'))
but4.place(x=5,y=322)

but5=Button(root,padx=5,pady=5,width=39,bg='white',fg='black',relief=GROOVE,command=all_gestures,text='Signs_Display',font=('helvetica 15 bold'))
but5.place(x=5,y=400)

but6=Button(root,padx=5,pady=5,width=5,bg='white',fg='black',relief=GROOVE,text='EXIT',command=root.destroy,font=('helvetica 15 bold'))
but6.place(x=210,y=478)

button1_window = my_canvas.create_window(5,176, anchor="nw", window=but2)
button2_window = my_canvas.create_window(5, 250, anchor="nw", window=but3)
button3_window = my_canvas.create_window(5, 322, anchor="nw", window=but4)
button4_window = my_canvas.create_window(5, 400, anchor="nw", window=but5)
button5_window = my_canvas.create_window(210, 478, anchor="nw", window=but6)


root.mainloop()

