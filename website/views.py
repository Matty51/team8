from website import app

from flask import render_template

@app.route('/')
@app.route('/home')
def home():
    return render_template('pages/home.html')

@app.route('/about')
def about():
    return render_template('pages/about.html')