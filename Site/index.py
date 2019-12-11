
# -*- coding: utf-8 -*-
import os
from flask import Flask, request,render_template,url_for
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
import sys
sys.path.insert(1, 'script')
from backend import model
print("path is equal to ",os.getcwd())
print("path i want is equal to ",os.path.realpath('images'))



app = Flask(__name__,static_url_path='/static')
app.config['UPLOADED_PHOTOS_DEST'] = os.path.realpath('images')



photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
patch_request_class(app)  

html = '''
    <!DOCTYPE html>
    <title>Upload File</title>
    <h1>Photo Upload</h1>
    <form method=post enctype=multipart/form-data>
         <input type=file name=photo>
         <input type=submit value=Upload>
    </form>
    '''

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST' and 'photo' in request.files:
        filename = photos.save(request.files['photo'])
        file_url = photos.url(filename)
        path,element = model(file_url)
        return render_template('index.html',image = file_url,label = element, results=path)
    return html


app.run(threaded=False)
