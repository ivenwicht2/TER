
# -*- coding: utf-8 -*-
from flask import Flask, render_template
app = Flask(__name__)


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

if __name__ == "__main__":
    app.run()
    
