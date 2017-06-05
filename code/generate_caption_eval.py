#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Sample code to generate caption using beam searcg using json file

'''
import sys
import json
import os
# comment out the below if you want to do type check. Remeber this have to be done BEFORE import chainer
# os.environ["CHAINER_TYPE_CHECK"] = "0"
import chainer 

import argparse
import numpy as np
import math
from chainer import cuda
import chainer.functions as F
from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers
from chainer import serializers

from CaptionGenerator import CaptionGenerator

#Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpu",default=-1, type=int, help=u"GPU ID.CPU is -1")
parser.add_argument('--eval_file',default='../data/MSCOCO/captions_val2014.json', type=str,help='evaluation file')
parser.add_argument('--image_feature_path',default='../data/MSCOCO/val2014_ResNet50_features/COCO_val2014_', type=str,help='path to the file of CNN features before image_id')
parser.add_argument('--filename_img_id',default=False,type=bool,help='image id is filename')
parser.add_argument('--cnn-model', type=str, default='../data/ResNet50.model',help='place of the ResNet model')
parser.add_argument('--rnn-model', type=str, default='../data/caption_en_model40.model',help='place of the caption model')
parser.add_argument('--vocab',default='../data/MSCOCO/mscoco_caption_train2014_processed_dic.json', type=str,help='path to the vocaburary json')
parser.add_argument('--beam',default=3, type=int,help='beam size in beam search')
parser.add_argument('--depth',default=50, type=int,help='depth limit in beam search')
parser.add_argument('--lang',default="<sos>", type=str,help='special word to indicate the langauge or just <sos>')
parser.add_argument('--output',default="../data/MSCOCO/val2014_predected_captions.json", type=str,help='output file name')
parser.add_argument("--hidden",default=512, type=int, help=u"number of hidden units in LSTM")
args = parser.parse_args()

caption_generator=CaptionGenerator(
    rnn_model_place=args.rnn_model,
    cnn_model_place=args.cnn_model,
    dictonary_place=args.vocab,
    beamsize=args.beam,
    depth_limit=args.depth,
    gpu_id=args.gpu,
    first_word= args.lang,
    hidden_dim=args.hidden,
    )

with open(args.eval_file, 'r') as f:
    captions = json.load(f)

image_feature_path_set=set()
for caption in captions["annotations"]:
    image_id = caption["image_id"]
    if args.filename_img_id:
        file_path="%s/%s.npz"%(args.image_feature_path,image_id)
    else:
        file_path="%s%012d.npz"%(args.image_feature_path,image_id)
    image_feature_path_set.add(file_path)

output_annotations={}
for i,fname in enumerate(image_feature_path_set):
    print(i, fname)
    image_feature=np.load(fname)['arr_0'].reshape(1,2048)
    captions=caption_generator.generate_from_img_feature(image_feature)
    output_annotations[fname] = captions
    # if i==101:
    #     break

with open(args.output, 'w') as f:
    json.dump(output_annotations, f, sort_keys=True, indent=4)
