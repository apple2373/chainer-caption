#!/usr/bin/env python
# -*- coding: utf-8 -*-

#python train_caption_model.py

import os
#os.environ["CHAINER_TYPE_CHECK"] = "0" #to disable type check. 
import chainer 
#Check che below is False if you disabled type check
#print(chainer.functions.Linear(1,1).type_check_enable) 

import argparse
import numpy as np
import chainer.functions as F
from chainer import cuda
from chainer import Function, FunctionSet, Variable, optimizers, serializers
from code.Image2CaptionDecoder import Image2CaptionDecoder
import json

#Parse arguments
parser = argparse.ArgumentParser(description=u"train caption generation model")
parser.add_argument("-g", "--gpu",default=-1, type=int, help=u"GPU ID.CPU is -1")
parser.add_argument("--savedir",default="./experiment1", type=str, help=u"The directory to save models and log")
parser.add_argument('--vocab',default='./data/MSCOCO/mscoco_caption_train2014_processed_dic.json', type=str,help='path to the vocaburary json')
parser.add_argument('--captions',default='./data/MSCOCO/mscoco_caption_train2014_processed.json', type=str,help='path to preprocessed caption json')
parser.add_argument('--img_features',default='./data/MSCOCO/train2014_ResNet50_features/', type=str,help='path to the directory containing CNN features')
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

with open(args.vocab, 'r') as f:
    index2token = json.load(f)

with open(args.captions, 'r') as f:
    captions = json.load(f)

#Model Preparation
print "preparing caption generation models and training process"
model=Image2CaptionDecoder(vocaburary_size=len(index2token))

#To GPU
if args.gpu >= 0:
    model.to_gpu()

#set up optimizers
optimizer = optimizers.Adam()
optimizer.setup(model)

#Trining Setting
batch_size=1
grad_clip = 1.0
num_train_data=len(captions)

#Start Training
print 'training started'
for epoch in xrange(1):
    print 'epoch %d' %epoch
    sum_loss = 0
    for caption_id in captions:
        optimizer.zero_grads()

        image_id=captions[caption_id]["image_id"]
        image_feature=np.load("%sCOCO_train2014_%012d.npz"%(args.img_features,image_id))['arr_0']
        image_feature=image_feature.reshape(-1,2048)
        x_batch=[xp.array(captions[caption_id]["token_ids"],dtype=np.int32)]
        if args.gpu >= 0:
            image_feature = cuda.to_gpu(image_feature, device=args.gpu)


        hx=xp.zeros((model.n_layers, batch_size, model.hidden_dim), dtype=xp.float32)
        cx=xp.zeros((model.n_layers, batch_size, model.hidden_dim), dtype=xp.float32)
        model.input_cnn_feature(hx,cx,image_feature)
        loss = model(hx, cx, x_batch)

        print loss.data
        with open(args.savedir+"/real_loss.txt", "a") as f:
            f.write(str(loss.data)+'\n') 

        loss.backward()
        optimizer.clip_grads(grad_clip)
        optimizer.update()
        
        sum_loss += loss.data * batch_size
    
    serializers.save_hdf5(args.savedir+"/caption_model%d.model"%epoch, model)
    serializers.save_hdf5(args.savedir+"/optimizer%d.model"%epoch, optimizer)

    mean_loss = sum_loss / num_train_data
    with open(args.savedir+"/mean_loss.txt", "a") as f:
        f.write(str(loss.data)+'\n')

