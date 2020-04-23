from flask import Blueprint, render_template,request,current_app
from . import db
import sys
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
sys.path.append("project/script/")
from backend import model_prediction
from PIL import Image
import base64
import numpy as np
import torch 
import pickle 
import io
from project import create_app
import os 
from werkzeug.utils import secure_filename

main = Blueprint('main', __name__)


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app = create_app()



device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.load("project/script/save/model").to(device)
with open('project/script/save/simi', 'rb') as handle:
    all_simi = pickle.load(handle)
with open('project/script/save/path_simi', 'rb') as handle:
    all_path = pickle.load(handle)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



@main.route('/')
def accueil():
    return render_template('accueil.html')

@main.route('/profile')
def profile():
    return render_template('profile.html')


@main.route('/Bibliotheque')
def biblio():
    return render_template('Bibliotheque.html')
    
@main.route('/Historique')
def histo():
    return render_template('Historique.html')
    
@main.route('/Labellisation')
def label():
    return render_template('Labellisation.html')


@main.route('/Apropos')
def ap():
    return render_template('apropos.html')

@main.route('/Analyse', methods=['GET', 'POST'])
def analyse():
    if request.method == 'POST':
        print("1")
        if 'file' not in request.files:
            return render_template('Analyse.html')
        file = request.files['file']
        print(file)
        if file and allowed_file(file.filename):
            print("2")
            filename = secure_filename(file.filename)
            filename = filename.replace('\\','/')
            file_url = os.path.join('project/images/', filename)
            file.save(file_url)
            
            origin = Image.open(file_url)
            file_object = io.BytesIO()
            origin.save(file_object, 'jpeg',quality=100)
            figdata_jgp = base64.b64encode(file_object.getvalue())
            origine_saved = figdata_jgp.decode('ascii')

            path,label,element = model_prediction(file_url,device,model,all_simi,all_path)
            result = []
            for el in path :
                img = Image.fromarray((el).astype(np.uint8))
                file_object = io.BytesIO()
                img.save(file_object, 'jpeg',quality=100)
                figdata_jgp = base64.b64encode(file_object.getvalue())
                result.append(figdata_jgp.decode('ascii'))
            return render_template('Analyse.html',image = origine_saved ,label = element, results=zip(result,label))
    return render_template('Analyse.html')