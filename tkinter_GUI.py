import  cv2
import tensorflow as tf
import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
from tkinter.filedialog import asksaveasfile
from PIL import ImageTk, Image
import matplotlib.pyplot as plt
import os
import time
# create the root window
from datetime import datetime
root = tk.Tk()
root.geometry('600x600')
root.resizable(False, False)

root.title('Tkinter Open File Dialog')
img_lr=[]

fn=''

images=[]
def select_file():
    global img_lr,images
    filetypes = (
        ('image files', '*.png'),
        ('image files', '*.jpg'),
        ('All files', '*.*')
    )

    filename = fd.askopenfilename(
        title='Open a file',
        initialdir='/',
        filetypes=filetypes)

    showinfo(
        title='Selected File',
        message=filename
    )
    fn=filename

    image1 = Image.open(fn)
    w,h=image1.size
    newsize = (128, 128)
    image2 = image1.resize(newsize)
    test = ImageTk.PhotoImage(image2)
    img_lr=image1
    label2['image'] = test
    label2.image = test
    img = cv2.imread(fn)
    images = []
    images.append(img)

img_sr=[]
img_bicubik=[]
a=0
def matpolib_open(image,id):
    plt.figure(id)
    imgplot = plt.imshow(image)
    plt.show()
def start():
    start_time = datetime.now()
    global fn,images,a,img_sr,img_bicubik
    srganPreGen=[]
    print(type(selected_size.get()))
    if selected_size.get()=='1':
        print(selected_size.get())
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        srganPreGen = tf.keras.models.load_model('bestSR.h5', compile=False)
    elif selected_size.get() == '2':
        print(selected_size.get())
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        srganPreGen = tf.keras.models.load_model('bestSR.h5', compile=False)

    images = np.array(images)
    srganPreGenPred = srganPreGen.predict(images)[0]
    srganPreGenPred = np.array(srganPreGenPred)
    img = cv2.cvtColor(srganPreGenPred, cv2.COLOR_BGR2RGB)
    a=srganPreGenPred
    im_pil = Image.fromarray((img * 1).astype(np.uint8)).convert('RGB')
    w, h = im_pil.size
    newsize = (128,128)
    im_pil1 = im_pil.resize(newsize)
    test = ImageTk.PhotoImage(im_pil1)
    label1['image']=test
    img_sr=im_pil
    label1.image = test
    label_org = ttk.Label(root,text="orginal image").grid(column=0,row=2)
    label_super = ttk.Label(root, text="ours image").grid(column=1, row=2)
    label_bucubik = ttk.Label(root, text="bicubic image").grid(column=3, row=2)
    print((images[0].shape))
    im_bic = cv2.cvtColor(images[0], cv2.COLOR_BGR2RGB)
    img_bicubik1= cv2.resize(im_bic,None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    img_bicubik1 = Image.fromarray((img_bicubik1 * 1).astype(np.uint8)).convert('RGB')

    im_pil1 = img_bicubik1.resize(newsize)
    img_bicubik=img_bicubik1
    test1 = ImageTk.PhotoImage(im_pil1)
    label3['image'] = test1
    label3.image = test1
    label5 = ttk.Label(text="Protses vaqti"+str(datetime.now() - start_time)[:len(str(datetime.now() - start_time))-6])
    label5.grid(column=3, row=3)



def save():
    files = [('All Files', '*.*'),
             ('image save', '*.jpg'),('image files', '*.png')]
    file = asksaveasfile(filetypes=files, defaultextension=files)
    print(file.name)
    cv2.imwrite(file.name, a)



# open button
open_button = ttk.Button(
    root,
    text='Open a File',
    command=select_file
).grid(column=0,row=0)
start_button = ttk.Button(
    root,
    text='Super_resolution',
    command=start
).grid(column=1,row=0)

save_button = ttk.Button(
    root,
    text='save as super_resolution',
    command= save
).grid(column=3,row=0)
sizes = (('GPU', 1),
         ('CPU', 2))
label1 = ttk.Button(root,command=lambda:matpolib_open(img_sr,1))
label1.grid(column=1,row=1)
label2 = ttk.Button(root, command=lambda: matpolib_open(img_lr, 0))
label2.grid(column=0, row=1)
label3 = ttk.Button(root, command=lambda: matpolib_open(img_bicubik, 2))
label3.grid(column=3, row=1)
label4 = ttk.Label(text="Protsesorni tanlang")
label4.grid(column=4, row=0)
selected_size = tk.StringVar()
for size in sizes:
    r = ttk.Radiobutton(
        root,
        text=size[0],
        value=size[1],
        variable=selected_size
    )
    r.grid(column=4, row=size[1])
root.mainloop()
