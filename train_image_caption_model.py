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
os.environ["CHAINER_TYPE_CHECK"] = "0" #to disable type check. 
import chainer 

import chainer.functions as F
from chainer import cuda
from chainer import Function, FunctionSet, Variable, optimizers, serializers

sys.path.append('./code')
from Image2CaptionDecoder import Image2CaptionDecoder
from CaptionDataLoader2 import CaptionDataLoader
from CaptionEvaluater import CaptionEvaluater
from CaptionGenerator import CaptionGenerator
from ResNet50 import ResNet

# from copy import deepcopy

#Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpu",default=-1, type=int, help=u"GPU ID.CPU is -1")
parser.add_argument("--savedir",default="./experiments/experiment1_cnn", type=str, help=u"The directory to save models and log")
parser.add_argument('--captions',default='./data/MSCOCO/mscoco_train2014_all_preprocessed.json', type=str,help='path to preprocessed caption json')
parser.add_argument('--image_root',default='./data/MSCOCO/MSCOCO_raw_images/', type=str,help='path to image directory')
parser.add_argument('--image_feature_root',default='./data/MSCOCO/MSCOCO_ResNet50_features/', type=str,help='path to CNN features directory')
parser.add_argument('--preload',default=False,type=bool,help='preload all image features onto RAM')
parser.add_argument("--epoch",default=60, type=int, help=u"the number of epochs")
parser.add_argument("--batch",default=128, type=int, help=u"mini batchsize")
parser.add_argument("--batch-cnn",default=16, type=int, help=u"mini batchsize when tuning cnn")
parser.add_argument("--hidden",default=512, type=int, help=u"number of hidden units in LSTM")
parser.add_argument("--cnn-tune-after",default=40, type=int, help=u"epoch starting to tune CNN. -1 means never")
parser.add_argument('--cnn-model', type=str, default='./data/ResNet50.model',help='place of the ResNet model')
parser.add_argument('--rnn-model', type=str, default='',help='place of the RNN model')
parser.add_argument('--depth',default=50, type=int,help='depth limit in beam search')
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

#make evaluater
evaluater=CaptionEvaluater()

#Prepare Data
print("loading preprocessed training data")

with open(args.captions, 'r') as f:
    captions = json.load(f)

val_truth={cap["file_path"]:cap["captions"] for cap in captions["val"]}
test_truth={cap["file_path"]:cap["captions"] for cap in captions["test"]}
evaluater.set_ground_truth(val_truth)
del captions["val"]
del captions["test"]
dataset=CaptionDataLoader(captions,image_feature_root=args.image_feature_root,image_root=args.image_root)

caption_generator=CaptionGenerator(
    rnn_model_place=args.rnn_model,
    cnn_model_place=args.cnn_model,
    dictonary_place=args.captions,
    beamsize=1,
    depth_limit=args.depth,
    gpu_id=args.gpu,
    first_word= "<sos>",
    )

#Model Preparation
print "preparing caption generation models and training process"
model=chainer.Chain()
model.rnn=Image2CaptionDecoder(vocaburary_size=len(captions["words"]),hidden_dim=args.hidden)
model.cnn=ResNet()
model.rnn.train=True
model.cnn.train=True
serializers.load_hdf5(args.cnn_model, model.cnn)
if not len(args.rnn_model) == 0:
    serializers.load_hdf5(args.rnn_model, model.rnn)

#To GPU
if args.gpu >= 0:
    model.cnn.to_gpu()
    model.rnn.to_gpu()

#set up optimizers
optimizer = optimizers.Adam()
optimizer.setup(args.rnn_model)
# optimizer_cnn = optimizers.Adam()
# optimizer_cnn.setup(model.cnn)

#Trining Setting
batch_size=args.batch
grad_clip = 1.0
num_train_data=len(captions)

#Start Training
print 'training started'

sum_loss = 0
print dataset.epoch
iteration = 1
evaluation_log={}
evaluation_log["val"]=[]
while (dataset.epoch <= args.epoch):
    optimizer.zero_grads()
    current_epoch=dataset.epoch
    train_cnn = current_epoch > args.cnn_tune_after and args.cnn_tune_after >= 0

    if train_cnn: 
        batch_size=args.batch_cnn
        #optimizer_cnn.zero_grads()
        images,x_batch=dataset.get_batch(batch_size,raw_image=True)
        if args.gpu >= 0:
            images = cuda.to_gpu(images, device=args.gpu)
            x_batch = [cuda.to_gpu(x, device=args.gpu) for x in x_batch]
        image_feature=model.cnn(images,t="feature")
    else:
        image_feature,x_batch=dataset.get_batch(batch_size)
        if args.gpu >= 0:
            image_feature = cuda.to_gpu(image_feature, device=args.gpu)
            x_batch = [cuda.to_gpu(x, device=args.gpu) for x in x_batch]

    #forward start
    hx=xp.zeros((model.rnn.n_layers, len(x_batch), model.rnn.hidden_dim), dtype=xp.float32)
    cx=xp.zeros((model.rnn.n_layers, len(x_batch), model.rnn.hidden_dim), dtype=xp.float32)
    hx,cx = model.rnn.input_cnn_feature(hx,cx,image_feature)
    loss = model.rnn(hx, cx, x_batch)

    print loss.data
    with open(args.savedir+"/real_loss.txt", "a") as f:
        f.write(str(loss.data)+'\n') 

    loss.backward()
    loss.unchain_backward()
    optimizer.clip_grads(grad_clip)
    optimizer.update()
    if train_cnn:
        pass 
        # optimizer_cnn.clip_grads(grad_clip)
        # optimizer_cnn.update()
    
    sum_loss += loss.data * batch_size
    iteration+=1
    
    if dataset.epoch - current_epoch > 0:
        print "epoch:",current_epoch
        if train_cnn: 
            serializers.save_hdf5(args.savedir+"/caption_model_resnet%d.model"%current_epoch, model.cnn)
        serializers.save_hdf5(args.savedir+"/caption_model%d.model"%current_epoch, model.rnn)
        serializers.save_hdf5(args.savedir+"/optimizer%d.model"%current_epoch, optimizer)

        mean_loss = sum_loss / num_train_data
        with open(args.savedir+"/mean_loss.txt", "a") as f:
            f.write(str(mean_loss)+'\n')
        sum_loss = 0
        iteration=0

        #evaluation
        print("evaluating for epoch %d"%current_epoch)
        caption_generator.cnn_model=model.cnn
        caption_generator.cnn_model.train=False
        caption_generator.rnn_model=model.rnn
        caption_generator.rnn_model.train=False

        val_predicted={}
        i=1
        for file_path in val_truth:
            i+=1
            sentence=" ".join(caption_generator.generate(args.image_root+"/"+file_path)[0]["sentence"][1:-1])
            print(i,sentence)
            val_predicted[file_path]=[sentence]

        print("computing evaluation scores")
        scores=evaluater.evaluate(val_predicted)
        print(scores)

        evaluation_log["val"].append(scores)
        with open(args.savedir+"/evaluation_log.json", "w") as f:
            json.dump(evaluation_log, f, sort_keys=True, indent=4)

        model.rnn.train=True
        model.cnn.train=True


#finalize the evaliation
best_epoch=np.argmax([score["cider"] for score in evaluation_log["val"]])+1#because epoch start from 1
print("best epoch is %d"%best_epoch)
if train_cnn: 
    serializers.load_hdf5(args.savedir+"/caption_model_resnet%d.model"%best_epoch, caption_generator.cnn_model)
serializers.load_hdf5(args.savedir+"/caption_model%d.model"%best_epoch, caption_generator.rnn_model)

caption_generator.cnn_model.train=False
caption_generator.rnn_model.train=False

test_predicted={}
i=1
for file_path in test_truth:
    i+=1
    sentence=" ".join(caption_generator.generate(args.image_root+"/"+file_path)[0]["sentence"][1:-1])
    print(i,sentence)
    test_predicted[file_path]=[sentence]

print("computing final evaluation scores")
evaluater.set_ground_truth(test_truth)
scores=evaluater.evaluate(test_predicted)
print(scores)

evaluation_log["test"]=scores
evaluation_log["best_epoch"]=best_epoch
with open(args.savedir+"/evaluation_log.json", "w") as f:
    json.dump(evaluation_log, f, sort_keys=True, indent=4)