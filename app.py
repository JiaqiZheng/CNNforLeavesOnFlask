# USAGE
# run this file
# 'http://localhost:5000'
# 

# import the necessary packages
from PIL import Image

import cv2

import numpy as np

from keras.models import load_model
from keras.preprocessing import image
import sys

import flask
from flask import render_template
from flask import Flask, request, redirect, url_for
import io
import os
from werkzeug.utils import secure_filename

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None



def loadmodel():
    img_width, img_height = 150, 150
    global model
    model = load_model('con3-model.h5')
    model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
    print("load success")
    
def load_image(img):
    img = image.load_img(img, target_size=(150, 150))

    img_tensor = image.img_to_array(img)                    
    img_tensor = np.expand_dims(img_tensor, axis=0)         
    img_tensor /= 255.                                     
    
    return img_tensor
#def prepare_image(image,target):
	# if the image mode is not RGB, convert it
	#if image.mode != "RGB":
	#	image = image.convert("RGB")

	#image = cv2.resize(image,(150,150))
	#img = np.reshape(image,[1,150,150,3])

	#return img



@app.route("/predict", methods=["POST"])
def predict():
	# ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST": 
        print("POST")  
        if request.files['file']:
            photo = request.files['file']
            in_memory_file = io.BytesIO()
            photo.save(in_memory_file)
            print(photo)

            #data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
            #print(data)
            #color_image_flag = 1
            #image = cv2.imdecode(data, color_image_flag)
            
            image = load_image(photo)
            #image = prepare_image(image,target=(150, 150))
            #print(image)
            
            preds = model.predict(image)
            #print(preds[0][0])
            
	#result ="sick-default"	
    if preds[0][0]>0.5:
        result = "sick"
    else:
        result = "health"
    return result

@app.route("/")
def display():
    print("load")
    return render_template("index.html")

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	
	loadmodel()
	app.run()
