import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="GCP-Shubham.json"

import argparse
from enum import Enum
import io
import json

from google.cloud import vision
from google.cloud.vision import types
from PIL import Image, ImageDraw


class FeatureType(Enum):
    PAGE = 1
    BLOCK = 2
    PARA = 3
    WORD = 4
    SYMBOL = 5

def get_annotated_result(json_vals):
    # Omitting this code from submission to prevent exposure to credentials. Working on another way
    return json_vals

def text_response_to_prediction(text_response):
    prediction = []
    for item in text_response:
      prediction.append(item[0])

    json_vals = {"%d" % (i+1): prediction[i] for i in range(len(prediction))}
    result = get_annotated_result(json_vals)
    return result

def get_response(image, bounds, client):
    im = image
    text_response = []

    for bound in bounds:
        im2 = im.crop([bound.vertices[0].x-2, bound.vertices[0].y-2,
                       bound.vertices[2].x+1, bound.vertices[2].y+1])
        im2.save('output-crop.jpg', 'JPEG')

        image_to_open = 'output-crop.jpg'

        with open(image_to_open, 'rb') as image_file:
          content = image_file.read()

        image = vision.types.Image(content=content)
        text_res = client.text_detection(image=image)

        texts = [text.description for text in text_res.text_annotations]
        text_response.append(texts)

    return text_response


def get_document_bounds(image_file, feature):
    """Returns document bounds given an image."""
    client = vision.ImageAnnotatorClient()

    bounds = []

    with io.open(image_file, 'rb') as image_file:
        content = image_file.read()

    image = types.Image(content=content)

    response = client.document_text_detection(image=image)
    document = response.full_text_annotation

    # Collect specified feature bounds by enumerating all document features
    for page in document.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    for symbol in word.symbols:
                        if (feature == FeatureType.SYMBOL):
                            bounds.append(symbol.bounding_box)

                    if (feature == FeatureType.WORD):
                        bounds.append(word.bounding_box)

                if (feature == FeatureType.PARA):
                    bounds.append(paragraph.bounding_box)

            if (feature == FeatureType.BLOCK):
                bounds.append(block.bounding_box)

    # The list `bounds` contains the coordinates of the bounding boxes.
    return bounds
