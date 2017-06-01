#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Sample code to generate caption using greedy search
'''
import sys
import json
import os
#comment out the below if you want to do type check. Remeber this have to be done BEFORE import chainer
# os.environ["CHAINER_TYPE_CHECK"] = "0" 
import chainer 

import argparse
import numpy as np
import math
from chainer import cuda
import chainer.functions as F
from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers
from chainer import serializers


sys.path.append('./code')
from image_loader import Image_loader
from ResNet50 import ResNet
from Image2CaptionDecoder import Image2CaptionDecoder

#Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpu",default=-1, type=int, help=u"GPU ID.CPU is -1")
parser.add_argument('--vocab',default='./data/MSCOCO/mscoco_caption_train2014_processed_dic.json', type=str,help='path to the vocaburary json')
parser.add_argument('--img',default='./sample_imgs/dog.jpg', type=str,help='path to the image')
parser.add_argument('--cnn-model', type=str, default='./data/ResNet50.model',help='place of the ResNet model')
parser.add_argument('--rnn-model', type=str, default='./data/caption_model.model',help='place of the caption model')
args = parser.parse_args()

image_loader=Image_loader(mean='imagenet')
with open(args.vocab, 'r') as f:
	token2index = json.load(f)
index2token={v:k for k,v in token2index.items()}

cnn_model=ResNet()
serializers.load_hdf5(args.cnn_model, cnn_model)
cnn_model.train = False
rnn_model=Image2CaptionDecoder(len(token2index))
serializers.load_hdf5(args.rnn_model, rnn_model)
rnn_model.train = False

if args.gpu >= 0:
	xp = cuda.cupy
	cuda.get_device(args.gpu).use()
	cnn_model.to_gpu()
	rnn_model.to_gpu()
else:
	xp=np

batch_size=1
hx=xp.zeros((rnn_model.n_layers, batch_size, rnn_model.hidden_dim), dtype=xp.float32)
cx=xp.zeros((rnn_model.n_layers, batch_size, rnn_model.hidden_dim), dtype=xp.float32)
img=image_loader.load(args.img)
if args.gpu >=0:
	img=cuda.to_gpu(img)
image_feature=cnn_model(img, "feature").data.reshape(1,1,2048)

hx,cx = rnn_model.input_cnn_feature(hx,cx,image_feature)
word=[xp.array([token2index["<sos>"]],dtype=xp.int32)]

for i in xrange(50):	
	hx, cx, word = rnn_model(hx, cx, word)
	word_idx=np.argmax(word[0].data)
	print(index2token[int(word_idx)], end=' ')
	word=[xp.array([word_idx],dtype=xp.int32)]
	if token2index["<eos>"]==word_idx:
		break

