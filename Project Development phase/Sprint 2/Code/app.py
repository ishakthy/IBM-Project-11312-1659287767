import requests
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from werkzeug.utils import secure_filename
from tensorflow.python.keras.backend import set_session
from flask import Flask, render_template, redirect, url_for, request


app = Flask(__name__)
model=load_model('ECG.h5')
@app.route('/')
def about():
    return render_template('about.html')

@app.route('/about')
def home():
    return render_template('about.html')

@app.route('/info')
def information():
    return render_template('info.html')

@app.route('/index')
def test():
    return render_template('index.html')

@app.route("/predict", methods=["GET", "POST"]) #route for our prediction 
def upload():
    if request.method== 'POST':
        f=request.files['image'] #requesting the file 
        basepath=os.path.dirname(__file__)#storing the file directory
        filepath=os.path.join(basepath, 'uploads',f.filename) #storing the file in uploads folder
        f.save(filepath) #saving the file

        img=image.load_img(filepath, target_size=(64,64)) #load and reshaping the image 
        x=image.img_to_array(img)#converting image to array 
        x=np.expand_dims(x, axis=0) #changing the dimensions of the image
        pred=model.predict_classes(x) #predicting classes 
        print("prediction", pred) #printing the prediction
        index=['Left Bundle Branch Block', 'Normal', 'Premature Atrial Contraction', 'Premature Ventricular Contractions', 'Right Bundle Branch Block', 'Ventricular Fibrillation'] 
        result=str(index[pred[0]])
        return result 
    
#port = int(os.getenv("PORT"))
if __name__=="__main__":
    app.run(debug=False) #running our app #app.run(host="0.0.0.0', port-port)