#!/usr/bin/python
# coding: UTF-8

#python ResNet50predict.py

import os
import numpy as np
import chainer
from chainer import cuda
from chainer import serializers, Variable
import chainer.functions as F
from ResNet50 import ResNet
from image_loader import Image_loader
import argparse

#parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--img', type=str, default="../sample_imgs/dog.jpg",help='place of a image that you want to predict')
parser.add_argument('--model', type=str, default='../data/ResNet50.model',help='place of the ResNet model')
parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()

#setup image loader
image_loader=Image_loader("imagenet")

#set up and load the model
model = ResNet()
serializers.load_hdf5(args.model, model)
model.train = False

#load image
img=image_loader.load(args.img)

#GPU preparation
if args.gpu >= 0:
	cuda.get_device(args.gpu).use()
	model.to_gpu()
	img = cuda.to_gpu(img, device=args.gpu)

#predict!
#img = Variable(img)
pred = F.softmax(model(img, None)).data

if args.gpu >= 0:
	pred = cuda.to_cpu(pred)

#print results
with open('../data/synset_words.txt') as f:
	synsets = f.read().split('\n')[:-1]

for i in np.argsort(pred)[0][-1::-1][:5]:
	print(synsets[i], pred[0][i])
