#!/usr/bin/python
# coding: UTF-8

#python ResNet_feature_extractor.py --img-dir ../data/MSCOCO/train2014 --out-dir ../data/MSCOCO/train2014_ResNet50_features
#python ResNet_feature_extractor.py --img-dir ../data/MSCOCO/val2014 --out-dir ../data/MSCOCO/val2014_ResNet50_features

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
parser.add_argument('--img-dir', type=str,help='The directory that the images are stored')
parser.add_argument('--out-dir', type=str,help='The directory that the features will be saved')
parser.add_argument('--model', type=str, default='../data/ResNet50.model',help='place of the ResNet model')
parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()

#setup image loader
image_loader=Image_loader(mean="imagenet")

#set up and load the model
model = ResNet()
serializers.load_hdf5(args.model, model)
model.train = False

#GPU preparation
if args.gpu >= 0:
	cuda.get_device(args.gpu).use()
	model.to_gpu()

image_files = os.listdir(args.img_dir)
i=0
for path in image_files:
	name, ext = os.path.splitext(path)
	print(i, path)
	img = image_loader.load(args.img_dir+'/'+path)
	if args.gpu >= 0:
		img = cuda.to_gpu(img, device=args.gpu)
	features = model(img, "feature").data
	if args.gpu >= 0:
		features = cuda.to_cpu(features)
	np.savez("%s/%s.npz"%(args.out_dir,name),features.reshape(2048))
	#np.load("%s/%s.npz"%(output_directory,name))['arr_0']
	i+=1
