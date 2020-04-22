from flask import Blueprint
from . import db

main = Blueprint('main', __name__)

@main.route('/profile')
def profile():
    return render_template('profile.html')