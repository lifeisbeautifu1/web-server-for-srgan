from flask import Flask, request, send_file
from flask_cors import cross_origin
from PIL import Image
from utils import allowed_file, serve_pil_image, upscale, need_preprocessing, process_jpg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)

@cross_origin(origins=['https://super-resolution-with-srgan.netlify.app'])
@app.route('/api/upload', methods = ['POST'])
def upload_file():
    if 'photo' not in request.files:
        return 'Forgot to send photo?', 400

    photo = request.files['photo']

    if photo.filename == '':
        return 'No selected file', 400
    
    if photo and allowed_file(photo.filename):
        # Do the magic.
        low_res_image = Image.open(photo)

        if need_preprocessing(photo.filename):
            low_res_image = process_jpg(low_res_image)

        high_res_image = upscale(low_res_image)
        return serve_pil_image(high_res_image)
    else:
        return "Only png's and jpg's are allowed", 400

@app.route('/')
def hello_world():
    return 'Hello, World!'
