
# -*- coding: utf-8 -*-
import os
from flask import Flask, request,render_template,url_for
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
import sys
sys.path.insert(1, 'script')
from backend import model
import io
from PIL import Image
import base64
import numpy as np




app = Flask(__name__)
app.config['UPLOADED_PHOTOS_DEST'] = os.path.realpath('images')



photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
patch_request_class(app)  

"""html = '''
    <!DOCTYPE html>
    <title>Upload File</title>
    <h1>Photo Upload</h1>
    <form method=post enctype=multipart/form-data>
         <input type=file name=photo>
         <input type=submit value=Upload>
    </form>
    '''"""
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST' and 'photo' in request.files:
        filename = photos.save(request.files['photo'])
        file_url = photos.url(filename)
        path,element = model(file_url)
        result = []
        for el in path :
            img = Image.fromarray((el * 255).astype(np.uint8))
            file_object = io.BytesIO()
            img.save(file_object, 'jpeg',quality=100)
            figdata_jgp = base64.b64encode(file_object.getvalue())
            result.append(figdata_jgp.decode('ascii'))
        return render_template('display.html',image = file_url,label = element, results=result)
    return render_template('index.html')


app.run(threaded=False)
render_template('index.html')
