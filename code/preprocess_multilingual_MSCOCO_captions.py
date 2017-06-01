#!/usr/bin/python
# coding: UTF-8

import os
import argparse
import json
import nltk
from collections import Counter,OrderedDict
from itertools import chain
from copy import deepcopy

import tinysegmenter
segmenter = tinysegmenter.TinySegmenter()

import random
random.seed(0)



'''
e.g. 

python preprocess_multilingual_MSCOCO_captions.py \
--en ../data/MSCOCO/captions_train2014.json \
--jp ../data/MSCOCO/yjcaptions26k_clean.json \
--outdir ../data/MSCOCO/ \
--prefix mscoco_caption_multi \

python preprocess_multilingual_MSCOCO_captions.py \
--en ../data/MSCOCO/captions_train2014.json \
--jp ../data/MSCOCO/yjcaptions26k_clean.json \
--outdir ../data/MSCOCO/ \
--prefix mscoco_caption_multi2 \
'''

def segment(caption,lang):
    caption_tokens=[lang]
    if lang=="<jp>":
        caption_tokens += segmenter.tokenize(caption)
    elif lang=="<en>":
        caption_tokens += nltk.word_tokenize(caption)
    caption_tokens.append("<eos>")

    return caption_tokens

def read_MSCOCO_json(file_place,args,lang):
        
    f = open(file_place, 'r')
    jsonData = json.load(f)
    f.close()

    captions={}
    #key:caption_id
    #image_id = captions[caption_id]['image_id']
    #caption = captions[caption_id]['tokens']#tokenized_caption

    for caption_data in jsonData['annotations']:
        caption_id=caption_data['id']
        image_id=caption_data['image_id']
        caption=caption_data['caption']

        caption=caption.replace('\n', '').strip().lower()
        if caption[-1]=='.':#to delete the last period. 
            caption=caption[0:-1]

        caption_tokens=segment(caption,lang)
        
        captions[caption_id]={}
        captions[caption_id]['image_id']=image_id
        captions[caption_id]['tokens']=caption_tokens
        captions[caption_id]['caption']=caption
        captions[caption_id]['id']=caption_id
        
    return captions


def create_new_caption_dataset(args,new_captions,new_cap_id=1):
    captions={}
    for caption in new_captions:
        captions[new_cap_id]=caption
        new_cap_id+=1

    #count word frequencies
    texts=[captions[caption_id]['tokens'] for caption_id in captions]
    tokens=list(chain.from_iterable(texts))
    freq_count=Counter(tokens)

    print("total distinct words:", len(freq_count))

    #remove words that appears less than 5
    id2word = [word for (word,freq) in freq_count.items() if freq >= args.cut]
    id2word.append("<ukn>")
    word2id = {id2word[i]:i for i in xrange(len(id2word))}

    print("total distinct words after cutoff:", len(id2word))

    for caption_id in captions:
        caption_tokens=captions[caption_id]['tokens']
        #map each token into id
        sentence=[]
        for token in caption_tokens:
            if token not in word2id:
                token="<ukn>"
            sentence.append(word2id[token])
        del captions[caption_id]['tokens']
        captions[caption_id]['token_ids']=sentence

    return captions,word2id,new_cap_id

def save_raw_json(captions,outfile,args):
    with open(args.en, 'r') as f:
        jsonData=json.load(f)
    captions={"annotations":captions,"images":jsonData['images'],"type":"captions","info":"dummy, not all image ids show up in captions","licenses":"dummy"}
    with open(outfile, 'w') as f:
        json.dump(captions, f, sort_keys=True, indent=4)    

def save_training_json_and_dic(captions,word2id,outfile,outdic):
    #save to json files
    with open(outfile, 'w') as f:
        json.dump(captions, f, sort_keys=True, indent=4)

    with open(outdic, 'w') as f:
        json.dump(word2id, f, sort_keys=True, indent=4)

if __name__ == '__main__':
    #parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--en', type=str,help='json file of MSCOCO formatted captions for English')
    parser.add_argument('--jp', type=str,help='json file of MSCOCO formatted captions for Japanese')
    parser.add_argument('--outdir', type=str, help='the directory of output files')
    parser.add_argument('--prefix', type=str, help='prefix of output files')
    parser.add_argument('--cut', default = 5,type=int,help='cut off frequency')
    args = parser.parse_args()

    #load jsonfile
    en_captions=read_MSCOCO_json(args.en,args,lang="<en>")
    jp_captions=read_MSCOCO_json(args.jp,args,lang="<jp>")

    en_img_ids=[caption["image_id"] for caption in en_captions.values()]
    jp_img_ids=[caption["image_id"] for caption in jp_captions.values()]

    common_img_ids = set(en_img_ids) & set(jp_img_ids)
    remain_img_ids = set(en_img_ids) - set(jp_img_ids)

    common_img_ids=list(common_img_ids)
    random.shuffle(common_img_ids)

    print("common image ids", len(common_img_ids))

    # en_remainings=[]
    en_selected_captions_train=[]
    en_selected_captions_val=[]
    en_selected_captions_test=[]
    jp_selected_captions_train=[]
    jp_selected_captions_val=[]
    jp_selected_captions_test=[]


    val_set=set(common_img_ids[0:2000])
    test_set=set(common_img_ids[2000:4000])
    train_set=set(common_img_ids[4000:-1])

    print(len(train_set))

    for caption_id, caption in en_captions.items():
        if caption["image_id"] in val_set:
            en_selected_captions_val.append(caption)
        elif caption["image_id"] in test_set:
            en_selected_captions_test.append(caption)
        elif caption["image_id"] in train_set:
            en_selected_captions_train.append(caption)

    for caption_id, caption in jp_captions.items():
        if caption["image_id"] in val_set:
            jp_selected_captions_val.append(caption)
        elif caption["image_id"] in test_set:
            jp_selected_captions_test.append(caption)
        elif caption["image_id"] in train_set:
            jp_selected_captions_train.append(caption)

    print("new caption created")

    save_raw_json(en_selected_captions_train,args.outdir+args.prefix+"_en_captions_train.json",args)
    save_raw_json(en_selected_captions_val,args.outdir+args.prefix+"_en_captions_val.json",args)
    save_raw_json(en_selected_captions_test,args.outdir+args.prefix+"_en_captions_test.json",args)
    save_raw_json(jp_selected_captions_train,args.outdir+args.prefix+"_jp_captions_train.json",args)
    save_raw_json(jp_selected_captions_val,args.outdir+args.prefix+"_jp_captions_val.json",args)
    save_raw_json(jp_selected_captions_test,args.outdir+args.prefix+"_jp_captions_test.json",args)

    new_captions=en_selected_captions_train+jp_selected_captions_train
    captions,word2id,new_cap_id = create_new_caption_dataset(args,deepcopy(new_captions),new_cap_id=1)
    outfile=args.outdir+args.prefix+"_en_jp_train_preprocessed.json"
    outdic=args.outdir+args.prefix+"_en_jp_train_dic.json"
    save_training_json_and_dic(captions,word2id,outfile,outdic)

    new_captions=en_selected_captions_train
    captions,word2id,new_cap_id = create_new_caption_dataset(args,deepcopy(new_captions),new_cap_id)
    outfile=args.outdir+args.prefix+"_en_train_preprocessed.json"
    outdic=args.outdir+args.prefix+"_en_train_dic.json"
    save_training_json_and_dic(captions,word2id,outfile,outdic)

    new_captions=jp_selected_captions_train
    captions,word2id,new_cap_id = create_new_caption_dataset(args,deepcopy(new_captions),new_cap_id)
    outfile=args.outdir+args.prefix+"_jp_train_preprocessed.json"
    outdic=args.outdir+args.prefix+"_jp_train_dic.json"
    save_training_json_and_dic(captions,word2id,outfile,outdic)

