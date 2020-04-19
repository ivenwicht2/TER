
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

@app.route('/Analyse')
def analyse():
    return render_template('Analyse.html')

@app.route('/Apropos')
def ap():
    return render_template('apropos.html')

if __name__ == "__main__":
    app.run()
    
