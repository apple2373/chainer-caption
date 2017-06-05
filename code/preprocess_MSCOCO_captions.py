#!/usr/bin/python
# coding: UTF-8

import os
import argparse
import json
import nltk
from collections import Counter,OrderedDict
from itertools import chain

'''
e.g. 
python preprocess_MSCOCO_captions.py \
--input ../data/MSCOCO/captions_train2014.json \
--output ../data/MSCOCO/mscoco_caption_train2014_processed.json \
--outdic ../data/MSCOCO/mscoco_caption_train2014_processed_dic.json \
--outfreq ../data/MSCOCO/mscoco_caption_train2014_processed_freq.json #this is just internal file
'''

def read_MSCOCO_json(file_place,args):
        
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

        caption_tokens=['<sos>']
        if args.char:
            caption_tokens += list(caption)
        else:
            caption_tokens += nltk.word_tokenize(caption)
        caption_tokens.append("<eos>")
        
        captions[caption_id]={}
        captions[caption_id]['image_id']=image_id
        captions[caption_id]['tokens']=caption_tokens
        
    return captions

if __name__ == '__main__':
    #parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str,help='json file of MSCOCO formatted captions')
    parser.add_argument('--output', type=str,help='the place of output file')
    parser.add_argument('--outdic', type=str,help='the place of output dictonary file')
    parser.add_argument('--outfreq', type=str,help='the place of output frequency map file')
    parser.add_argument('--char', default = False,type=bool,help='character based tokenization. e.g. for Japanese')
    parser.add_argument('--cut', default = 5,type=int,help='cut off frequency')
    args = parser.parse_args()

    #load jsonfile
    captions=read_MSCOCO_json(args.input,args)

    #count word frequencies
    texts=[captions[caption_id]['tokens'] for caption_id in captions]
    tokens=list(chain.from_iterable(texts))
    freq_count=Counter(tokens)

    # freq_count_ordered={}
    # for word,freq in freq_count.most_common(len(freq_count)):
    #     freq_count_ordered[word]=freq
    #頻度順にjsonを出したいけど、諦めたw　valueでorderするのはできない？

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

    #save to json files
    with open(args.output, 'w') as f:
        json.dump(captions, f, sort_keys=True, indent=4)

    with open(args.outdic, 'w') as f:
        json.dump(word2id, f, sort_keys=True, indent=4)

    if args.outfreq != None:
        with open(args.outfreq, 'w') as f:
            json.dump(freq_count, f, sort_keys=True, indent=4)