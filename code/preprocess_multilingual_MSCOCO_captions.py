#!/usr/bin/python
# coding: UTF-8

import os
import argparse
import json
import nltk
from collections import Counter,OrderedDict
from itertools import chain
import tinysegmenter
segmenter = tinysegmenter.TinySegmenter()

import random
random.seed(0)


'''
e.g. 

python preprocess_multilingual_MSCOCO_captions.py \
--en ../data/MSCOCO/captions_train2014.json \
--jp ../data/MSCOCO/yjcaptions26k_clean.json \
--out_train ../data/MSCOCO/mscoco_caption_en_jp_train.json \
--out_val ../data/MSCOCO/mscoco_caption_en_jp_val.json \
--out_test ../data/MSCOCO/mscoco_caption_en_jp_test.json \
--outdic ../data/MSCOCO/mscoco_caption_en_jp_train_dic.json \

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
        
    return captions

if __name__ == '__main__':
    #parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--en', type=str,help='json file of MSCOCO formatted captions for English')
    parser.add_argument('--jp', type=str,help='json file of MSCOCO formatted captions for Japanese')
    parser.add_argument('--out_train', type=str, help='the place of output training file')
    parser.add_argument('--out_val', type=str, help='the place of output validitation file')
    parser.add_argument('--out_test', default = False,type=bool,help='the place of output test file')
    parser.add_argument('--outdic', type=str,help='the place of output dictonary file')
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

    print "common image ids",len(common_img_ids)

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
    for caption_id, caption in en_captions.iteritems():
        if caption["image_id"] in val_set:
            en_selected_captions_val.append(caption)
        elif caption["image_id"] in test_set:
            en_selected_captions_test.append(caption)
        elif caption["image_id"] in train_set:
            en_selected_captions_train.append(caption)

    for caption_id, caption in jp_captions.iteritems():
        if caption["image_id"] in val_set:
            jp_selected_captions_val.append(caption)
        elif caption["image_id"] in test_set:
            jp_selected_captions_test.append(caption)
        elif caption["image_id"] in train_set:
            jp_selected_captions_train.append(caption)

    # for caption_id, caption in en_captions.iteritems():
    #     if caption["image_id"] in common_img_ids[0:2000]:
    #         en_selected_captions_val.append(caption)
    #     elif caption["image_id"] in common_img_ids[2000:4000]:
    #         en_selected_captions_test.append(caption)
    #     elif caption["image_id"] in common_img_ids[4000:-1]:
    #         en_selected_captions_train.append(caption)

    # for caption_id, caption in jp_captions.iteritems():
    #     if caption["image_id"] in common_img_ids[0:2000]:
    #         jp_selected_captions_val.append(caption)
    #     elif caption["image_id"] in common_img_ids[2000:4000]:
    #         jp_selected_captions_test.append(caption)
    #     elif caption["image_id"] in common_img_ids[4000:-1]:
    #         jp_selected_captions_train.append(caption)

    print "new caption created"

    new_cap_id=1
    captions={}
    for caption in en_selected_captions_train+jp_selected_captions_train:
        captions[new_cap_id]=caption
        new_cap_id+=1

    #count word frequencies
    texts=[captions[caption_id]['tokens'] for caption_id in captions]
    tokens=list(chain.from_iterable(texts))
    freq_count=Counter(tokens)

    # freq_count_ordered={}
    # for word,freq in freq_count.most_common(len(freq_count)):
    #     freq_count_ordered[word]=freq
    #頻度順にjsonを出したいけど、諦めたw　valueでorderするのはできない？

    print "total distinct words:",len(freq_count)

    #remove words that appears less than 5
    id2word = [word for (word,freq) in freq_count.iteritems() if freq >= args.cut]
    id2word.append("<ukn>")
    word2id = {id2word[i]:i for i in xrange(len(id2word))}

    print "total distinct words after cutoff:",len(id2word)

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

    #save to json files
    with open(args.out_train, 'w') as f:
        json.dump(captions, f, sort_keys=True, indent=4)

    with open(args.outdic, 'w') as f:
        json.dump(word2id, f, sort_keys=True, indent=4)

    # with open(args.out_train, 'w') as f:
    #     json.dump(captions, f, sort_keys=True, indent=4)

    # with open(args.out_train, 'w') as f:
    #     json.dump(captions, f, sort_keys=True, indent=4)
    # if args.outfreq != None:
    #     with open(args.outfreq, 'w') as f:
    #         json.dump(freq_count, f, sort_keys=True, indent=4)