import os
from flask import Flask, request
import json
from flask_cors import CORS
import base64
import onxxmodel

app = Flask(__name__)
cors = CORS(app)
datasetPath = 'data'

@app.route('/testing', methods=['POST'])
def result():
    data = json.loads(request.data.decode('utf-8'))
    print(type(data))
    image = data['image'].split(',')[1].encode('utf-8')
    os.makedirs('data2', exist_ok=True)
    filename = 'image.png'
    with open(f'userdata/{filename}', 'wb') as f:
        f.write(base64.decodebytes(image))
    output = onxxmodel.test(f'userdata/{filename}')
    print(output)
    return output

@app.route('/upload_canvas', methods = ['POST'])
def upload_canvas():
    data = json.loads(request.data.decode('utf-8'))
    image_data = data['image'].split(',')[1].encode('utf-8')
    fileName = data['filename']
    className = data['className']
    os.makedirs(f'{datasetPath}/{className}/image',exist_ok = True)
    with open(f'{datasetPath}/{className}/image/{fileName}',"wb") as fh:
        fh.write(base64.decodebytes(image_data))

    return "Got the image"

if __name__ == "__main__":
    app.run()