#!/usr/bin/env python
# -*- coding: utf-8 -*-

#class to keep several CaptionDataLoader
#Each caption loader is expcted to output raw captions instead of indicies 
#because this loader holds the golbal dictonary

import numpy as np
import os
import json
from CaptionDataLoader2 import CaptionDataLoader


class CaptionMultiDataLoader(object):
    def __init__(self,datasets_paths,image_feature_root,image_root,preload):
        self.num_captions=0
        all_captions_dic={}#dictonary to keep the info about several langs
        all_words=set()#collet all words so that we can create new dic
        #test code
        for lang_and_json in datasets_paths.split(";")[0:-1]:
            lang,json_file=lang_and_json.split(":")
            all_captions_dic[lang]={}
            all_captions_dic[lang]["json"]=json_file
            with open(json_file, 'r') as f:
                captions = json.load(f)
            self.num_captions+=len(captions["captions"])
            #put the evaluation data into  all_captions_dic[lang]
            all_captions_dic[lang]["val"]={cap["file_path"]:cap["captions"] for cap in captions["val"]}
            all_captions_dic[lang]["test"]={cap["file_path"]:cap["captions"] for cap in captions["test"]}
            index2word={ word['idx']:word['word'] for word in captions["words"]}
            for caption in captions["captions"]:
                caption["caption"] = [index2word[word] for word in caption["caption"][1:]]
            all_words=all_words.union(index2word.values())
            all_words.add(lang)
            del captions["val"]
            del captions["test"]
            del captions["words"]
            all_captions_dic[lang]["dataset"]=CaptionDataLoader(captions,image_feature_root=image_feature_root,image_root=image_root,preload_all_features=preload,holding_raw_captions=True)

        #first create a new dic
        if "<sos>" in all_words:
            all_words.remove("<sos>")
        self.word2index={word:idx for idx,word in enumerate(list(all_words))}

        #all_captions_dic is defined in the train_image_caption_model
        self.all_captions_dic=all_captions_dic
        #initialize epoch changed flag
        self.epoch_changed={}
        for lang in self.all_captions_dic.keys():
            self.epoch_changed[lang]=False
        self.epoch=1
        self.num_langs=len(self.all_captions_dic.keys())

    def get_batch(self,batch_size,raw_image=False):
        #just get several batches from all data_loaders
        batch_size=int(batch_size/self.num_langs)
        batch_word_indices=[]
        imgs_list=[]
        for lang,dataset_dic in self.all_captions_dic.items():
            dataset=dataset_dic["dataset"]
            current_epoch=dataset.epoch
            image_features,x_batch=dataset.get_batch(batch_size,raw_image)
            imgs_list.append(image_features)
            #this is just converting words into indicies and make it in a chainer training compatible way
            batch_word_indices+=[np.array([self.word2index[word] for word in [lang]+sent],dtype=np.int32) for sent in x_batch]
            #if this lang dataset entered the new epoch, mark it.
            if dataset.epoch - current_epoch > 0:
                self.epoch_changed[lang]=True
        batch_images=np.vstack(imgs_list)

        if all(self.epoch_changed.values()):
            self.epoch+=1
            for lang,dataset_dic in self.all_captions_dic.items():
                dataset=dataset_dic["dataset"]
                dataset.suffle_data()
                dataset.index_count=0

        return batch_images,batch_word_indices

if __name__ == '__main__':
    datasets_paths="<en>:../data/MSCOCO/mscoco_train2014_all_preprocessed.json;<jp>:../data/MSCOCO/mscoco_jp_all_preprocessed.json;"
    image_feature_root="../data/MSCOCO/MSCOCO_ResNet50_features/"
    image_root="../data/MSCOCO/MSCOCO_raw_images/"
    dataset=CaptionMultiDataLoader(datasets_paths,image_feature_root,image_root)
    batch_image_features,batch_word_indices =  dataset.get_batch(10)
    for batch in batch_word_indices:
        print(batch)
    print(dataset.word2index["<jp>"],dataset.word2index["<en>"])
    print(batch_image_features.shape)
    
    batch_images,batch_word_indices =  dataset.get_batch(10,raw_image=True)
    for batch in batch_word_indices:
        print(batch)
    print(dataset.word2index["<jp>"],dataset.word2index["<en>"])
    print(batch_images.shape)
