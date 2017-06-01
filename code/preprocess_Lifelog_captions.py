#!/usr/bin/python
# coding: UTF-8

import sys
import os
import argparse
import json
import nltk
from collections import Counter,OrderedDict
from itertools import chain
from copy import deepcopy
import numpy as np

import tinysegmenter
segmenter = tinysegmenter.TinySegmenter()

import random
random.seed(0)



'''
e.g. 

python preprocess_Lifelog_captions.py \
--input ../data/Lifelog/amt_captions.txt \
--outdir ../data/Lifelog/ \
--prefix Lifelog 


'''
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
    captions={"annotations":captions,"images":"fake","type":"captions","info":"dummy, not all image ids show up in captions","licenses":"dummy"}
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
    parser.add_argument('--input', type=str,help='text file of lifelog captions')
    parser.add_argument('--outdir', type=str, help='the directory of output files')
    parser.add_argument('--prefix', type=str, help='prefix of output files')
    parser.add_argument('--cut', default = 0,type=int,help='cut off frequency')
    args = parser.parse_args()


    f = open(args.input)
    line = f.readline() # 1行を文字列として読み込む(改行文字も含まれる)
    
    im2captions={}
    while line:
        line = f.readline().strip()
        line_array=line.split(" ")
        img_id=line_array[0][0:-4]

        if not os.path.exists("../data/Lifelog/amt_images_ResNet50Features/%s.npz"%img_id):
            continue

        caption = [ word.lower().replace(".","") for word in line_array[1:] ]
        if len(caption)==0:
            continue
        if "i" in caption:
            caption=["<first>"]+caption+["<eos>"]
        else:
            caption=["<third>"]+caption+["<eos>"]

        caption={
        "image_id":img_id,
        "caption":" ".join(caption),
        "tokens":caption,
        }

        if img_id not in im2captions:
            im2captions[img_id]=[caption]
        else:
            im2captions[img_id].append(caption)
    f.close()
    img_ids=im2captions.keys()

    print(img_ids)

    captions_train=[]
    captions_val=[]

    val_set=set(img_ids[0:10])
    train_set=set(img_ids[10:])

    print(len(train_set))

    for img_id, captions in im2captions.items():
        for caption in captions:
            if img_id in val_set:
                captions_val.append(caption)
            elif img_id in train_set:
                captions_train.append(caption)

    for img_id in val_set:
        print(img_id)
        for cap in im2captions[img_id]:
            print(cap["caption"])

    # sys.exit()

    print("new caption created")

    save_raw_json(captions_train,args.outdir+args.prefix+"_captions_train.json",args)
    save_raw_json(captions_val,args.outdir+args.prefix+"_captions_val.json",args)

    captions,word2id,new_cap_id = create_new_caption_dataset(args,deepcopy(captions_train),new_cap_id=1)
    outfile=args.outdir+args.prefix+"_train_preprocessed.json"
    outdic=args.outdir+args.prefix+"_train_dic.json"
    save_training_json_and_dic(captions,word2id,outfile,outdic)
