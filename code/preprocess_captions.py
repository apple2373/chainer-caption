#!/usr/bin/python
# coding: UTF-8
#script for a new preprocess way.
#input file is [{ "file_path": "path/img.jpg", "captions": ["a caption", "a similar caption" ...] }, ...]

import os
import argparse
import json
from collections import Counter,OrderedDict
from itertools import chain
import random


'''
e.g. 
python preprocess_captions.py \
--input ../data/MSCOCO/ms_coco_raw.json \
--output ../data/MSCOCO/mscoco_train2014_all_preprocessed.json \
'''

if __name__ == '__main__':
    #parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str,help='json file of MSCOCO formatted captions')
    parser.add_argument('--output', type=str,help='the place of output file')
    parser.add_argument('--char', default = False,type=bool,help='character based tokenization. e.g. for Japanese')
    parser.add_argument('--jp', default = False,type=bool,help='use Japanese segmenter. Install by "pip install tinysegmenter"')
    parser.add_argument('--cn', default = False,type=bool,help='use Chinese segmenter. Install by "pip install jieba"')
    parser.add_argument('--cut', default = 5,type=int,help='cut off frequency. this frequency will be the lowest to be kept.')
    parser.add_argument('--lower', default = True,type=bool,help='make everything into lower case')
    parser.add_argument('--remove-period', default = True,type=bool,help='remove the last period if a caption has a priod')
    parser.add_argument('--val', default = 2000,type=int,help='number of validiation images')
    parser.add_argument('--test', default = 2000,type=int,help='number of test images')
    # parser.add_argument('--keep-info', default = False,type=bool,help='keep other image information other than "file_path" in the input json')
    args = parser.parse_args()

    class Segmenter:
        def __init__(self, args):
            self.args=args
            if self.args.jp:
                import tinysegmenter
                self.tinysegmenter = tinysegmenter.TinySegmenter()
                self.segmenter= lambda sent: self.tinysegmenter.tokenize(sent)
            elif self.args.cn:
                import jieba
                jieba.cut("我来到北京清华大")
                self.segmenter= lambda sent: list(jieba.cut(sent))
            elif not args.char:
                import nltk
                self.nltk=nltk
                self.segmenter = lambda sent: self.nltk.word_tokenize(sent)
            else:
                self.segmenter= lambda sent: list(sent)

        def segment(self,caption):
            if self.args.lower:
                caption=caption.replace('\n', '').strip().lower()
            if self.args.remove_period and caption[-1]=='.':
                caption=caption[0:-1]#to delete the last period. 
            return self.segmenter(caption) 

    def word2idx_func(word,word2idx):
        if word in word2idx:
            return word2idx[word]
        else:
            return word2idx["<ukn>"]

    segmenter=Segmenter(args)

    with open(args.input, 'r') as f:
        jsonData = json.load(f)

    #split into train,val,test data:
    train_data=[]
    val_data=[]
    test_data=[]

    random.seed(0)#in order to have the same split every time
    random.shuffle(jsonData)

    for i,img in enumerate(jsonData):
        if i < args.val:
            new_captions=[]
            for caption in img["captions"]:
                new_cap=" ".join(segmenter.segment(caption.lower()))
                new_captions.append(new_cap)
            img["captions"]=new_captions
            val_data.append(img)
        elif i < args.test + args.val: 
            new_captions=[]
            for caption in img["captions"]:
                new_cap=" ".join(segmenter.segment(caption.lower()))
                new_captions.append(new_cap)
            img["captions"]=new_captions
            test_data.append(img)
        else:
            train_data.append(img)

    #the info for the output json
    image_idx=0
    caption_idx=0
    all_images=[]
    all_captions=[]
    all_words=[]

    for image in train_data:
        for caption in image["captions"]:
            caption_tokens=['<sos>']
            caption_tokens += segmenter.segment(caption)
            caption_tokens.append("<eos>")
            all_captions.append({"image_idx":image_idx,"caption":caption_tokens,"idx":caption_idx})
            caption_idx+=1
        del image["captions"]
        all_images.append({"file_path":image["file_path"],"idx":image_idx}) 
        image_idx+=1

    #count word frequencies
    tokens=list(chain.from_iterable([caption["caption"] for caption in  all_captions]))
    freq_count=Counter(tokens)

    print("total distinct words:",len(freq_count))
    print("top 20 frequent words:")
    for word,freq in freq_count.most_common(20):
        print(u"(%s,%s)"%(word,freq))

    #remove words that appears less than the cut
    ukn_freq=0
    for word,freq in freq_count.most_common():
        if freq >= args.cut:
            all_words.append({"word":word,"freq":freq,"idx":len(all_words)})
        else:
            ukn_freq+=freq
    all_words.append({"word":"<ukn>","freq":ukn_freq,"idx":len(all_words)})

    print("total distinct words after the cutoff:",len(all_words))

    word2idx = {word["word"]:word["idx"] for word in all_words}

    for caption in all_captions:
        caption["caption"] = [word2idx_func(word,word2idx) for word in caption["caption"]]

    #save to json files
    preprocessed_file={}
    preprocessed_file["images"]=all_images
    preprocessed_file["captions"]=all_captions
    preprocessed_file["words"]=all_words
    preprocessed_file["val"]=val_data
    preprocessed_file["test"]=test_data
    with open(args.output, 'w') as f:
        json.dump(preprocessed_file, f, sort_keys=True, indent=4)
    print("The output file is saved to",args.output)