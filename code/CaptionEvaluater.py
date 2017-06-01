#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
evaluate captions by Blue, Rouge, and Cider 

This can be used for any langauge assumning that the inputs are already segmented (i.e. tokenized). 
The ground truth should be lowered before passing to here..
'''

import sys
sys.path.append('../coco-caption/')
sys.path.append('../coco-caption/pycocoevalcap/')
sys.path.append('./coco-caption/')
sys.path.append('./coco-caption/pycocoevalcap/')
from bleu.bleu import Bleu
from rouge.rouge import Rouge
from cider.cider import Cider
import string
import re

class CaptionEvaluater(object):
    def __init__(self,):
        self.blue_scorer=Bleu(4)
        self.rouge_scorer=Rouge()
        self.cider_scorer=Cider()
        self.truth=None
        remove = string.punctuation+"、。，．"
        self.remove_pattern = r"[{}]".format(remove) # create the pattern

    def remove_punctuation(self,line):
        #I am not sure how unicode works in python, so just in case.
        line=line.replace(u"<unk>","")
        line=line.replace("<unk>","")
        line=line.replace(u"。","")
        line=line.replace('\u3002',"")
        return re.sub(self.remove_pattern, "", line) 

    def trnasform_utf8(self,line):
        # return u' '.join(line).encode('utf-8').strip()
        return line

    def set_ground_truth(self,ground_truth):
        '''
        ground_truth should be a python dictonary whose shape is; 
            {"image_identifier": ["a caption", "a similar caption", ...], ...}
        "image_identifier" can be either string or number.
        '''
        for img in ground_truth:
            # ground_truth[img]=map(self.trnasform_utf8,ground_truth[img])
            ground_truth[img]=map(self.remove_punctuation,ground_truth[img])
        self.truth=ground_truth

    def evaluate(self,predicetd_captions):
        '''
        predicetd_captions should be a python dictonary whose shape is; 
            {"image_identifier": ["the prediced caption"], ...}
        "image_identifier" need to be same as used in ground truth.
        make sure the number of caption is only one, even though it uses python list. 
        '''
        for img in predicetd_captions:
            # predicetd_captions[img]=map(self.trnasform_utf8,predicetd_captions[img])
            predicetd_captions[img]=map(self.remove_punctuation,predicetd_captions[img])

        results={}
        for i,score in enumerate(self.get_bleu(predicetd_captions)[0]):
            results["bleu-%d"%i]=score
        results["rouge"] = self.get_rouge(predicetd_captions)[0]
        results["cider"] = self.get_cider(predicetd_captions)[0]

        return results

    def get_bleu(self,predicetd_captions):
        score, scores = self.blue_scorer.compute_score(self.truth, predicetd_captions) 
        #output is a python list [bleu-1,bleu-2,bleu-3,bleu-4]
        return score, scores

    def get_rouge(self,predicetd_captions):
        score, scores = self.rouge_scorer.compute_score(self.truth, predicetd_captions)
        return score, scores

    def get_cider(self,predicetd_captions):
        score, scores = self.cider_scorer.compute_score(self.truth, predicetd_captions)
        return score, scores

if __name__ == '__main__':
    #test
    #spaceで区切ったのを入れればOK.
    ground_truth={}
    ground_truth['262148'] = ['オレンジ色 の シャツ を 着た 人 が います',
      'オレンジ色 の Tシャツ を 着ている 人 が 立って います',
      '人 が オレンジ色 の シャツ を 着て 立って います',
    ]
    ground_truth[262148] = ['the skateboarder is putting on a show using the picnic table as his stage',
    'a skateboarder pulling tricks on top of a picnic table',
    'a man riding on a skateboard on top of a table',
    'a skate boarder doing a trick on a picnic table',
    'a person is riding a skateboard on a picnic table with a crowd watching']
    ground_truth[393225]= ['a bowl of soup that has some carrots shrimp and noodles in it',
    'the healthy food is in the bowl and ready to eat',
    'soup has carrots and shrimp in it as it sits next to chopsticks',
    'a tasty bowl of ramen is served for someone to enjoy',
    'bowl of asian noodle soup with shrimp and carrots']
    ground_truth[1] = ['the skateboarder is putting on a show using the picnic table as his stage',
    'a skateboarder pulling tricks on top of a picnic table',
    'a man riding on a skateboard on top of a table',
    'a skate boarder doing a trick on a picnic table',
    'a person is riding a skateboard on a picnic table with a crowd watching']
    ground_truth[2]= ['a bowl of soup that has some carrots shrimp and noodles in it',
    'the healthy food is in the bowl and ready to eat',
    'soup has carrots and shrimp in it as it sits next to chopsticks',
    'a tasty bowl of ramen is served for someone to enjoy',
    'bowl of asian noodle soup with shrimp and carrots']


    #prediceted は一つだけじゃないとダメ
    predicted={}
    predicted['262148']=['人 が オレンジ色 の シャツ を 着て 立って <unk> います。']
    predicted[262148]=['A man riding a skateboard down a ramp。']
    predicted[393225]=['A bowl of soup with carrots and a spoon.']
    predicted[1]=['a man riding a skateboard down a ramp。']
    predicted[2]=['a bowl of soup with carrots and a spoon、<unk>']
    #keyは数字でも文字列でもどっちでもいいけど、ground truth と predicedで対応が取れるように！

    evaluater=CaptionEvaluater()
    evaluater.set_ground_truth(ground_truth)
    print(evaluater.evaluate(predicted))
    #https://github.com/tylin/coco-caption/issues/5
    #Yes, CIDEr can have values till 10 (technically).