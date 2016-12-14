#!/usr/bin/python
# coding: UTF-8

import os
import numpy as np
from chainer import serializers, Variable
import chainer.functions as F
from ResNet50 import ResNet
from image_loader import Image_loader

image_loader=Image_loader("imagenet")

model = ResNet()
serializers.load_hdf5('../data/ResNet50.model', model)
model.train = False
# pred = model(x_data, "feature").data

img=image_loader.load("../sample_imgs/dog.jpg")
x_data = Variable(img)

pred = F.softmax(model(x_data, None)).data

with open('../data/synset_words.txt') as f:
	synsets = f.read().split('\n')[:-1]

for i in np.argsort(pred)[0][-1::-1][:5]:
    print synsets[i],pred[0][i]
