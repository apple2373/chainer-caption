#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
training code including fine tuning CNN
'''

import argparse
import numpy as np
import json

import sys
import os
#os.environ["CHAINER_TYPE_CHECK"] = "0" #to disable type check. 
import chainer 

import chainer.functions as F
from chainer import cuda
from chainer import Function, FunctionSet, Variable, optimizers, serializers

sys.path.append('./code')
from Image2CaptionDecoder import Image2CaptionDecoder
from CaptionDataLoader2 import CaptionDataLoader
from ResNet50 import ResNet


#Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpu",default=-1, type=int, help=u"GPU ID.CPU is -1")
parser.add_argument("--savedir",default="./experiment1_cnn", type=str, help=u"The directory to save models and log")
parser.add_argument('--captions',default='./data/MSCOCO/mscoco_train2014_all_preprocessed.json', type=str,help='path to preprocessed caption json')
parser.add_argument('--image_root',default='./data/MSCOCO/MSCOCO_raw_images/', type=str,help='path to image directory')
parser.add_argument('--image_feature_root',default='./data/MSCOCO/MSCOCO_ResNet50_features/', type=str,help='path to CNN features directory')
parser.add_argument('--preload',default=False,type=bool,help='preload all image features onto RAM')
parser.add_argument("--epoch",default=40, type=int, help=u"the number of epochs")
parser.add_argument("--batch",default=16, type=int, help=u"mini batchsize")
parser.add_argument("--hidden",default=512, type=int, help=u"number of hidden units in LSTM")
parser.add_argument("--cnn-tune-after",default=10, type=int, help=u"epoch starting to tune CNN")
parser.add_argument('--cnn-model', type=str, default='./data/ResNet50.model',help='place of the ResNet model')
parser.add_argument('--rnn-model', type=str, default='',help='place of the RNN model')
args = parser.parse_args()


#save dir
if not os.path.isdir(args.savedir):
    os.makedirs(args.savedir)
    print "made the save directory",args.savedir


#Gpu Setting
if args.gpu >= 0:
    xp = cuda.cupy 
    cuda.get_device(args.gpu).use()
else:
    xp=np

#Prepare Data
print("loading preprocessed training data")


with open(args.captions, 'r') as f:
    captions = json.load(f)
dataset=CaptionDataLoader(captions,image_feature_root=args.image_feature_root,image_root=args.image_root)


#Model Preparation
print "preparing caption generation models and training process"
model=chainer.Chain()
model.rnn_model=Image2CaptionDecoder(vocaburary_size=len(captions["words"]),hidden_dim=args.hidden)
model.cnn_model=ResNet()
serializers.load_hdf5(args.cnn_model, model.cnn_model)
if not len(args.rnn_model) == 0:
    serializers.load_hdf5(args.rnn_model, model.rnn_model)

#To GPU
if args.gpu >= 0:
    model.cnn_model.to_gpu()
    model.rnn_model.to_gpu()

#set up optimizers
optimizer = optimizers.Adam()
optimizer.setup(model)

#Trining Setting
batch_size=args.batch
grad_clip = 1.0
num_train_data=len(captions)

#Start Training
print 'training started'

sum_loss = 0
print dataset.epoch
iterraton = 1
while (dataset.epoch <= args.epoch):
    optimizer.zero_grads()
    current_epoch=dataset.epoch

    if current_epoch > args.cnn_tune_after: 
        images,x_batch=dataset.get_batch(batch_size,raw_image=True)
        if args.gpu >= 0:
            images = cuda.to_gpu(images, device=args.gpu)
            x_batch = [cuda.to_gpu(x, device=args.gpu) for x in x_batch]
        image_feature=model.cnn_model(images,t="feature")
    else:
        image_feature,x_batch=dataset.get_batch(batch_size)
        if args.gpu >= 0:
            image_feature = cuda.to_gpu(image_feature, device=args.gpu)
            x_batch = [cuda.to_gpu(x, device=args.gpu) for x in x_batch]

   

    #forward start
    hx=xp.zeros((model.rnn_model.n_layers, len(x_batch), model.rnn_model.hidden_dim), dtype=xp.float32)
    cx=xp.zeros((model.rnn_model.n_layers, len(x_batch), model.rnn_model.hidden_dim), dtype=xp.float32)
    hx,cx = model.rnn_model.input_cnn_feature(hx,cx,image_feature)
    loss = model.rnn_model(hx, cx, x_batch)

    print loss.data
    with open(args.savedir+"/real_loss.txt", "a") as f:
        f.write(str(loss.data)+'\n') 

    loss.backward()
    loss.unchain_backward()
    optimizer.clip_grads(grad_clip)
    optimizer.update()
    
    sum_loss += loss.data * batch_size
    iterraton+=1
    
    if dataset.epoch - current_epoch > 0 or iterraton > 10 :
        print "epoch:",dataset.epoch
        if current_epoch > args.cnn_tune_after: 
            serializers.save_hdf5(args.savedir+"/caption_model_resnet%d.model"%current_epoch, model.cnn_model)
        serializers.save_hdf5(args.savedir+"/caption_model%d.model"%current_epoch, model.rnn_model)
        serializers.save_hdf5(args.savedir+"/optimizer%d.model"%current_epoch, optimizer)

        mean_loss = sum_loss / num_train_data
        with open(args.savedir+"/mean_loss.txt", "a") as f:
            f.write(str(mean_loss)+'\n')
        sum_loss = 0
        iterraton=0