from pyexpat import model
import re
from flask import Flask, redirect, render_template, request, url_for
import cv2
from matplotlib import image
import numpy as np
app = Flask(__name__)

@app.route('/')
def home():
    return redirect(url_for('index'))

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        # handle image upload and processing logic here
        # set image_filename variable to the uploaded image filename
        image = request.files['image']
        image_filename = image.filename
        # do something with the image file
        return render_template('index.html', image_filename=image_filename)
    else:
        return render_template('index.html')
    
@app.route('/prediction')
def make_prediction(image_fp):
    def prediction(image_fp):
        im = cv2.imread(image_fp) # load image
        img = image.load_img(image_fp, target_size = (256,256))
        img = image.img_to_array(img)

        image_array = img / 255. # scale the image
        img_batch = np.expand_dims(image_array, axis = 0)
        
        class_ = ["Alluvail", "BlackSoil", "RedSoil"] # possible output values
        predicted_value = class_[model.predict(img_batch).argmax()]
        true_value = re.search(r'(Alluvial)|(BlackSoil)|(RedSoil)', image_fp)[0]
        
        out = f"""Predicted Soil Type: {predicted_value}
        True Soil Type: {true_value}
        Correct?: {predicted_value == true_value}"""
        
        return out

    test_image_filepath = 'train' + r'/Black/0.jpg'
    pred = prediction(test_image_filepath)
    message = ""
    if(pred=='Alluvial'):
        message = "paddy, wheat, maize and barley; cotton; groundnut, chillies, tobacco and pulses; sugarcane; oilseeds, vegetables and fruits"
    elif(pred=='BlackSoil'):
        message = "crops suitable for black soil include leguminous crops such as cotton, turn and citrus fruits, as well as tobacco, chilly and oil seeds."
    else:
        message = "Black pepper, Brinjal, Potatoes, Bean, Baby corn"
        
    return render_template('suggestion.html', message = message, pred = pred)

if __name__ == '__main__':
    app.run(debug=True)
