#!/usr/bin/env python
# -*- coding: utf-8 -*-

#class to get the data in a batch way
#loading on memory option (preload_all_features) took  6m10.400s (user time = 2m41.546s) to load  if it is true
#refactered version of CaptionDataLoader.py

import numpy as np
import os
from image_loader import Image_loader
from ResNet50 import ResNet

class CaptionDataLoader(object):
    def __init__(self,dataset,image_feature_root,image_root="",preload_all_features=False,image_mean="imagenet",holding_raw_captions=False):
        self.holding_raw_captions=holding_raw_captions
        self.image_loader=Image_loader(mean=image_mean)
        self.captions=dataset["captions"]
        self.num_captions=len(self.captions)
        self.images=dataset["images"]
        self.caption2image={caption["idx"]:caption["image_idx"] for caption in dataset["captions"]}
        self.image_feature_root=image_feature_root+"/"#path to preprocessed image features. It assume the feature are stored with the same name but only extension is changed to .npz
        self.image_root=image_root+"/"#path to image directory
        self.random_indicies = np.random.permutation(len(self.captions))
        self.index_count=0
        self.epoch=1
        self.preload_all_features=preload_all_features
        if  self.preload_all_features:
            self.image_features=np.array([np.load("%s/%s.npz"%(self.image_feature_root, os.path.splitext(image["file_path"])[0] ))['arr_0'] for image in self.images])

    def get_batch(self,batch_size,raw_image=False):
        #if raw_image is true, it will give you Batchx3x224x224 otherwise it will be just features
        batch_caption_indicies=self.random_indicies[self.index_count:self.index_count+batch_size]
        self.index_count+=batch_size
        if self.index_count > len(self.captions):
            self.epoch+=1
            self.suffle_data()
            self.index_count=0

        #sorry the following lines are so complicated...
        #this is just loading preprocessed images or image features and captions for this batch
        if raw_image:
            batch_images= np.array( [self.image_loader.load(self.image_root+self.images[self.caption2image[i]]["file_path"],expand_batch_dim=False) for i in batch_caption_indicies] )
        else:
            if self.preload_all_features:
                batch_images=self.image_features[[self.caption2image[i] for i in batch_caption_indicies]]
            else:
                batch_images=np.array([np.load("%s/%s.npz"%(self.image_feature_root, os.path.splitext(self.images[self.caption2image[i]]["file_path"])[0] ))['arr_0'] for i in batch_caption_indicies])
        if self.holding_raw_captions:
            batch_word_indices=[self.captions[i]["caption"] for i in batch_caption_indicies]
        else:
            batch_word_indices=[np.array(self.captions[i]["caption"],dtype=np.int32) for i in batch_caption_indicies]

        return batch_images,batch_word_indices

    def suffle_data(self):
        self.random_indicies = np.random.permutation(len(self.captions))


if __name__ == '__main__':
    #test code
    import json
    with open("../data/MSCOCO/mscoco_train2014_all_preprocessed.json", 'r') as f:
        captions = json.load(f)
    dataset=CaptionDataLoader(captions,image_feature_root="../data/MSCOCO/MSCOCO_ResNet50_features/",image_root="../data/MSCOCO/MSCOCO_raw_images/")
    batch_images,batch_word_indices =  dataset.get_batch(10,raw_image=True)
    print(batch_word_indices)
    print(batch_images)

    batch_image_features,batch_word_indices =  dataset.get_batch(10)
    print(batch_word_indices)
    print(batch_image_features.shape)

    dataset=CaptionDataLoader(captions,image_feature_root="../data/MSCOCO/MSCOCO_ResNet50_features",preload_all_features=True)
    batch_image_features,batch_word_indices =  dataset.get_batch(10)
    print(batch_word_indices)
    print(batch_image_features.shape)
