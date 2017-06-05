#!/usr/bin/env python
# -*- coding: utf-8 -*-

#class to get the data in a batch way
#loading on memory option (preload_all_features) took  6m10.400s (user time = 2m41.546s) to load  if it is true

import numpy as np

class CaptionDataLoader(object):
    def __init__(self, captions,image_feature_path,preload_all_features=False,filename_img_id=False):
        self.captions = captions
        self.image_feature_path=image_feature_path#path before image id. e.g. ../data/MSCOCO/train2014_ResNet50_features/COCO_train2014_
        self.caption_ids = captions.keys()
        self.random_indicies = np.random.permutation(len(self.captions))
        self.index_count=0
        self.epoch=1
        self.preload_all_features=preload_all_features
        self.filename_img_id=filename_img_id
        if  self.preload_all_features:
            if self.filename_img_id:
                self.image_features=np.array([np.load("%s/%s.npz"%(self.image_feature_path,self.captions[caption_id]["image_id"]))['arr_0'] for caption_id in self.caption_ids])
            else:
                self.image_features=np.array([np.load("%s%012d.npz"%(self.image_feature_path,self.captions[caption_id]["image_id"]))['arr_0'] for caption_id in self.caption_ids])

    def get_batch(self,batch_size):
        batch_data_indicies=self.random_indicies[self.index_count:self.index_count+batch_size]
        self.index_count+=batch_size
        if self.index_count > len(self.captions):
            self.epoch+=1
            self.suffle_data()
            self.index_count=0

        #sorry the following lines are so complicated...
        #this is just loading preprocessed images features and captions for this batch
        if self.preload_all_features:
            batch_image_features=self.image_features[batch_data_indicies]
        else:
            if self.filename_img_id:
                batch_image_features=np.array([np.load("%s/%s.npz"%(self.image_feature_path,self.captions[self.caption_ids[i]]["image_id"]))['arr_0'] for i in batch_data_indicies])
            else:
                batch_image_features=np.array([np.load("%s%012d.npz"%(self.image_feature_path,self.captions[self.caption_ids[i]]["image_id"]))['arr_0'] for i in batch_data_indicies])
        
        batch_word_indices=[np.array(self.captions[self.caption_ids[i]]["token_ids"],dtype=np.int32) for i in batch_data_indicies]

        return batch_image_features,batch_word_indices

    def suffle_data(self):
        self.random_indicies = np.random.permutation(len(self.captions))


if __name__ == '__main__':
    #test code
    import json
    with open("../data/MSCOCO/mscoco_caption_train2014_processed.json", 'r') as f:
        captions = json.load(f)
    dataset=CaptionDataLoader(captions,image_feature_path="../data/MSCOCO/train2014_ResNet50_features/COCO_train2014_")
    batch_image_features,batch_word_indices =  dataset.get_batch(10)
    print(batch_word_indices)
    print(batch_image_features.shape)

    dataset=CaptionDataLoader(captions,image_feature_path="../data/MSCOCO/train2014_ResNet50_features/COCO_train2014_",preload_all_features=True)
    batch_image_features,batch_word_indices =  dataset.get_batch(10)
    print(batch_word_indices)
    print(batch_image_features.shape)
