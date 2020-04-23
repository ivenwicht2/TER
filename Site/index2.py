
# -*- coding: utf-8 -*-
import os
from flask import Flask, request,render_template,url_for
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
import sys
#sys.path.insert(1, 'script')
from backend import model_prediction
import io
from PIL import Image
import base64
import numpy as np
import itertools
import torch 
import pickle 

device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.load("script/save/model").to(device)
with open('script/save/simi', 'rb') as handle:
    all_simi = pickle.load(handle)
with open('script/save/path_simi', 'rb') as handle:
    all_path = pickle.load(handle)



app = Flask(__name__)
app.config['UPLOADED_PHOTOS_DEST'] = os.path.realpath('images')

photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
patch_request_class(app)  

@app.route('/')
def accueil():
    return render_template('Accueil.html')
    
@app.route('/Bibliotheque')
def biblio():
    return render_template('Bibliotheque.html')
    
@app.route('/Historique')
def histo():
    return render_template('Historique.html')
    
@app.route('/Labellisation')
def label():
    return render_template('Labellisation.html')


@app.route('/Apropos')
def ap():
    return render_template('apropos.html')

@app.route('/Analyse', methods=['GET', 'POST'])
def analyse():
    if request.method == 'POST' and 'photo' in request.files:
        filename = photos.save(request.files['photo'])
        file_url = photos.url(filename)
        path,label,element = model_prediction(file_url,device,model,all_simi,all_path)
        result = []
        for el in path :
            img = Image.fromarray((el).astype(np.uint8))
            file_object = io.BytesIO()
            img.save(file_object, 'jpeg',quality=100)
            figdata_jgp = base64.b64encode(file_object.getvalue())
            result.append(figdata_jgp.decode('ascii'))
        return render_template('Analyse.html',image = file_url,label = element, results=zip(result,label))
    return render_template('Analyse.html')

if __name__ == "__main__":
    app.run()
    
