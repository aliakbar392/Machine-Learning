from starlette.applications import Starlette

from starlette.responses import JSONResponse, HTMLResponse, RedirectResponse

from fastai.vision import *

import torch

from pathlib import Path

from io import BytesIO

import sys

import uvicorn

import aiohttp

import asyncio
import numpy as np
from flask import Flask, redirect, url_for
app = Starlette()

# You path where you have stored models/weights.pth
path = Path(r'C:\Users\wasay\Desktop\Main Internshala Project\dataset-resized\models')
path2=Path(r'C:\Users\wasay\Desktop\Main Internshala Project\dataset-resized')

# Classes
classes = ['cardboard','glass','metal','paper','plastic','trash']
typo=['bio','non_bio','non_bio','bio','both','both']
lul = dict(zip(classes,typo))

# Create a DataBunch
np.random.seed(42)
data = ImageDataBunch.from_folder(path2, train=".", size=224, valid_pct=0.2,
        ds_tfms=get_transforms(),num_workers=4).normalize(imagenet_stats)

# Create a learner and load the weights
learn = cnn_learner(data, models.resnet34, metrics=accuracy)
learn.load('Main_Model_Garbage')



    

@app.route("/")
def form(request):
    return HTMLResponse("""    <body style="background-color:powderblue;">
    <img src="bg01.jpg" alt="Flowers in Chania">
        <h3><b><u>Garbage Classifier</u></b></h3>
            <form action="/upload" method="post" enctype="multipart/form-data">
            Select image to upload:
            <input type="file" name="file">
            <input type="submit" value="Upload Image">
        </form>
"""
	
    )


@app.route("/upload", methods=["POST"])
async def upload(request):
    data = await request.form()
    bytes = await (data["file"].read())
    return predict_image_from_bytes(bytes)

async def get_bytes(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()

def predict_image_from_bytes(bytes):
    img = open_image(BytesIO(bytes))
    _, class_, losses = learn.predict(img)



    return JSONResponse({
        "prediction": classes[class_.item()],
        


        "scores": sorted(
            zip(learn.data.classes, map(float, losses)),
            key=lambda p: p[1],
            reverse=True),
        "king":lul
       
        
    })

if __name__ == "__main__":
    if "serve" in sys.argv:
        uvicorn.run(app, host="0.0.0.0", port=80)
