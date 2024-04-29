from flask import Flask
app = Flask(__name__)

@app.route('/upload', methods = ['GET', 'POST'])
def upload_file():
    return 'Hi!'

@app.route('/')
def hello_world():
    return 'Hello, World!'
