#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
training code
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
from CaptionDataLoader import CaptionDataLoader

#Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpu",default=-1, type=int, help=u"GPU ID.CPU is -1")
parser.add_argument("--savedir",default="./experiment1", type=str, help=u"The directory to save models and log")
parser.add_argument('--vocab',default='./data/MSCOCO/mscoco_caption_train2014_processed_dic.json', type=str,help='path to the vocaburary json')
parser.add_argument('--captions',default='./data/MSCOCO/mscoco_caption_train2014_processed.json', type=str,help='path to preprocessed caption json')
parser.add_argument('--image_feature_path',default='./data/MSCOCO/train2014_ResNet50_features/COCO_train2014_', type=str,help='path to the file of CNN features before image_id')
parser.add_argument('--filename_img_id',default=False,type=bool,help='image id is filename')
parser.add_argument('--preload',default=False,type=bool,help='preload all image features onto RAM')
parser.add_argument("--epoch",default=10, type=int, help=u"the number of epochs")
parser.add_argument("--batch",default=128, type=int, help=u"mini batchsize")
parser.add_argument("--hidden",default=512, type=int, help=u"number of hidden units in LSTM")

args = parser.parse_args()

#save dir
if not os.path.isdir(args.savedir):
    os.makedirs(args.savedir)
    print("made the save directory", args.savedir)

#Gpu Setting
if args.gpu >= 0:
    xp = cuda.cupy 
    cuda.get_device(args.gpu).use()
else:
    xp=np

#Prepare Data
print("loading preprocessed training data")

with open(args.vocab, 'r') as f:
    index2token = json.load(f)

with open(args.captions, 'r') as f:
    captions = json.load(f)

dataset=CaptionDataLoader(captions,image_feature_path=args.image_feature_path,preload_all_features=args.preload, filename_img_id=args.filename_img_id)

#Model Preparation
print("preparing caption generation models and training process")
model=Image2CaptionDecoder(vocaburary_size=len(index2token),hidden_dim=args.hidden)

#To GPU
if args.gpu >= 0:
    model.to_gpu()

#set up optimizers
optimizer = optimizers.Adam()
optimizer.setup(model)

#Trining Setting
batch_size=args.batch
grad_clip = 1.0
num_train_data=len(captions)

#Start Training
print('training started')

sum_loss = 0
print(dataset.epoch)
iterraton = 1
while (dataset.epoch <= args.epoch):
    optimizer.zero_grads()
    current_epoch=dataset.epoch
    image_feature,x_batch=dataset.get_batch(batch_size)

    if args.gpu >= 0:
        image_feature = cuda.to_gpu(image_feature, device=args.gpu)
        x_batch = [cuda.to_gpu(x, device=args.gpu) for x in x_batch]

    hx=xp.zeros((model.n_layers, len(x_batch), model.hidden_dim), dtype=xp.float32)
    cx=xp.zeros((model.n_layers, len(x_batch), model.hidden_dim), dtype=xp.float32)
    hx,cx = model.input_cnn_feature(hx,cx,image_feature)
    loss = model(hx, cx, x_batch)

    print(loss.data)
    with open(args.savedir+"/real_loss.txt", "a") as f:
        f.write(str(loss.data)+'\n') 

    loss.backward()
    loss.unchain_backward()
    optimizer.clip_grads(grad_clip)
    optimizer.update()
    
    sum_loss += loss.data * batch_size
    iterraton+=1
    
    if dataset.epoch - current_epoch > 0 or iterraton > 10000:
        print("epoch:", dataset.epoch)
        serializers.save_hdf5(args.savedir+"/caption_model%d.model"%current_epoch, model)
        serializers.save_hdf5(args.savedir+"/optimizer%d.model"%current_epoch, optimizer)

        mean_loss = sum_loss / num_train_data
        with open(args.savedir+"/mean_loss.txt", "a") as f:
            f.write(str(mean_loss)+'\n')
        sum_loss = 0
        iterraton=0