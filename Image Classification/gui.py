import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy as np
from keras.models import load_model

# Load the trained model to classify the images
try:
    model = load_model('model.h5')
    print('Loaded an already existing model\nLaunching UI')
except:
    print('Saved Model not found. Please train the model first by running train.py')
    exit()

# dictionary to label all the CIFAR-10 dataset classes.
classes = {0: 'Aeroplane', 1: 'Automobile', 2: 'Bird', 3: 'Cat', 4: 'Deer', 5: 'Dog', 6: 'Frog', 7: 'Horse', 8: 'Ship',
           9: 'Truck'}

# initialise GUI
top = tk.Tk()
top.geometry('800x600')
top.title('Image Classification using CIFAR10 dataset')
top.configure(background='#CDCDCD')
label = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)


def classify(file_path):
    image = Image.open(file_path)
    image = image.resize((32, 32))
    image = np.expand_dims(image, axis=0)
    image = np.array(image)
    pred = np.argmax(model.predict([image])[0], axis=-1)
    sign = classes[int(pred)]
    print('Uploaded image is a:', sign)
    label.configure(foreground='#011638', text=sign)


def show_classify_button(file_path):
    classify_b = Button(top, text="Classify Image", command=lambda: classify(file_path), padx=10, pady=5)
    classify_b.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
    classify_b.place(relx=0.79, rely=0.46)


def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass


upload = Button(top, text="Browse files", command=upload_image, padx=10, pady=5)

upload.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))

upload.pack(side=BOTTOM, pady=50)
sign_image.pack(side=BOTTOM, expand=True)
label.pack(side=BOTTOM, expand=True)

heading = Label(top, text='Image Classification using CIFAR10 dataset', pady=20, font=('arial', 20, 'bold'))
heading.configure(background='#CDCDCD', foreground='#364156')
heading.pack()

# heading1 = Label(top, text='Hi', pady=20, font=('arial', 20, 'bold'))
# heading1.configure(background='#CDCDCD', foreground='#364156')
# heading1.pack()
top.mainloop()
