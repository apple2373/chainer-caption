#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Sample code to generate caption
'''
import numpy as np
from image_reader import Image_reader
from caption_generator import Caption_generator

#Instantiate image_reader with GoogleNet mean image
mean_image = np.array([104, 117, 123]).reshape((3,1,1))
image_reader=Image_reader(mean=mean_image)

#Instantiate caption generator
caption_model_place='../models/caption_model.chainer'
cnn_model_place='../data/bvlc_googlenet_caffe_chainer.pkl'
index2word_place='../work/index2token.pkl'
caption_generator=Caption_generator(caption_model_place=caption_model_place,cnn_model_place=cnn_model_place,index2word_place=index2word_place)


#The preparation is done
#Let's ganarate caption for a image

#First, read an image as numpy array
image_file_path='../images/test_image.jpg'
image=image_reader.read(image_file_path)


#Next, put the image into caption generator
#The output structure is 
#	[caption,caption,caption,...]
#	caption = {"sentence":This is a generated sentence, "probability": The probability of the generated sentence} 
captions=caption_generator.generate(image)

#For example, if you want to print all captions
for caption in captions:
    sentence=caption['sentence']
    probability=caption['probability']
    print " ".join(sentence),probability

#Let's do for another image
image_file_path='../images/COCO_val2014_000000241747.jpg'
image=image_reader.read(image_file_path)
captions=caption_generator.generate(image)
for caption in captions:
    sentence=caption['sentence']
    probability=caption['probability']
    print " ".join(sentence),probability
