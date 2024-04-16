#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 14:04:25 2024

@author: chloebright
"""
'''
import pandas as pd 
import matplotlib.pyplot as plt 
from transformers import pipeline 
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from datasets import load_metric
import numpy as np 
from mlscript import Tester
'''
from flask import Flask, render_template, request
from mlscript import predictor 


app = Flask(__name__)

@app.route("/")
@app.route("/index.html")
def home():
    return render_template("index.html")

@app.route("/about.html")
def about():
    return render_template("about.html")

@app.route("/contact.html")
def contact():
    return render_template("contact.html")

@app.route("/test.html")
def test():
    return render_template("test.html")

@app.route("/results.html", methods=["POST", "GET"])
def result():
    if request.method == "POST":
        news = str(request.form.get("news","")) #default ot empty string if not provided 
        results=predictor()
        results = predictor().predict(news)      
        return render_template("results.html", results=results)
    else: 
        return render_template("index.html")




if __name__ == "__main__": #checking if __name__'s value is '__main__'. __name__ is an python environment variable who's value will always be '__main__' till this is the first instatnce of app.py running
    app.run(debug=True,port=4949) #running flask (Initalized on line 19)
