import os
from flask import Flask, request, redirect, url_for, render_template
from flask_table import Table, Col
from werkzeug.utils import secure_filename

app = Flask(__name__)

import matplotlib.pyplot as plt
import numpy as np
import json
from pprint import pprint
from invoice import *

class ItemTable(Table):
    classes = ["table-bordered", "text-light"]
    name = Col('Entity')
    description = Col('Value')

class Item(object):
    def __init__(self, name, description):
        self.name = name
        self.description = description


@app.route('/', methods=['GET', 'POST'])
def main_page():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        # *** If input is in PDF format ***
        # from pdf2image import convert_from_path
        # pages = convert_from_path(filename)
        # for page in pages:
        #   page.save(filename[:-4] + '.jpg', 'JPEG')
        file.save(os.path.join('uploads', filename))
        return redirect(url_for('prediction', filename=filename))
    return render_template('index.html')

@app.route('/prediction/<filename>')
def prediction(filename):
    #Step 1
    img_path = os.path.join('uploads', filename)
    # my_image = plt.imread(os.path.join('uploads', filename))

    #Step 2
    bounds = get_document_bounds(img_path, FeatureType.PARA)

    image_file = Image.open(img_path)
    client = vision.ImageAnnotatorClient()
    client = vision.ImageAnnotatorClient.from_service_account_file(
    'GCP-Shubham.json'
    )

    # Step 3
    text_response = get_response(image_file,bounds,client)

    # Step 4
    while([] in text_response) : 
        text_response.remove([]) 

    result = text_response_to_prediction(text_response)    

    # Declare your table    
    items = []
    for value, name in result.items():
        items.append(Item(value, name))

    table = ItemTable(items)

    #Step 5
    return render_template('predict.html', table=table)
    

app.run(host='0.0.0.0', port=8000)

