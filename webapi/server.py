# coding: utf-8

#Author: Satoshi Tsutsui
#consluted to: http://qiita.com/tanikawa/items/a0ecf10638f327f63f3e
#consluted: http://bibouroku.viratube.com/2015/11/07/postされた画像データをflask上で受け取るには/


from __future__ import print_function
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify


import sys
import json
import sys

from datetime import datetime

import os
#comment out the below if you want to do type check. Remeber this have to be done BEFORE import chainer
#os.environ["CHAINER_TYPE_CHECK"] = "0" 
import chainer 
#If the below is false, the type check is disabled. 
#print(chainer.functions.Linear(1,1).type_check_enable) 

import cv2
import argparse
import numpy as np
import math
from chainer import cuda
import chainer.functions as F
from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers
from chainer import serializers

sys.path.append('../code/')
from CaptionGenerator import CaptionGenerator
from image_loader import Image_loader

image_loader=Image_loader(mean="imaganet")

app = Flask(__name__)

# app.config['UPLOAD_FOLDER'] = '/home/stsutsui/chainer-caption/webapi/uploads'

# JSON 中の日本語を ASCII コードに変換しないようにする (curl コマンドで見やすくするため。ASCII に変換しても特に問題ない)
app.config['JSON_AS_ASCII'] = False

# This is the path to the upload directory
app.config['UPLOAD_FOLDER'] = 'uploads/'
# These are the extension that we are accepting to be uploaded
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg', 'gif'])

# For a given file, return whether it's an allowed type or not
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('./index.html')

# 画像分類 API
# http://localhost:8090/predict に画像を投げると JSON で結果が返る
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    image = cv2.imdecode(np.fromstring(file.stream.read(), np.uint8), cv2.IMREAD_COLOR)
    time=str(datetime.now()).replace(" ","_")
    fname="uploads/"+time+'.jpg'
    cv2.imwrite(fname, image)
    captions=caption_generator.generate(fname)
     # 結果を JSON にして返す
    return jsonify({
        'captions': captions
    })

if __name__ == '__main__':
    #Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu",default=-1, type=int, help=u"GPU ID.CPU is -1")
    parser.add_argument('--vocab',default='./data/MSCOCO/mscoco_caption_train2014_processed_dic.json', type=str,help='path to the vocaburary json')
    parser.add_argument('--cnn-model', type=str, default='./data/ResNet50.model',help='place of the ResNet model')
    parser.add_argument('--rnn-model', type=str, default='./data/caption_model.model',help='place of the caption model')
    parser.add_argument('--beam',default=5, type=int,help='beam size in beam search')
    parser.add_argument('--depth',default=50, type=int,help='depth limit in beam search')
    args = parser.parse_args()

    caption_generator=CaptionGenerator(
        rnn_model_place=args.rnn_model,\
        cnn_model_place=args.cnn_model,\
        dictonary_place=args.vocab,\
        beamsize=args.beam,\
        depth_limit=args.depth,\
        gpu_id=args.gpu,)

    app.run(host='0.0.0.0', port=8090)