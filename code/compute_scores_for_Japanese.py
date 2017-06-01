#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
sys.path.append('../coco-caption/')
sys.path.append('../coco-caption/pycocoevalcap/')
from bleu.bleu import Bleu
from rouge.rouge import Rouge
from cider.cider import Cider


#spaceで区切ったのを入れればOK.
ground_truth={}
ground_truth['262148'] = ['オレンジ色 の シャツ を 着た 人 が います',
  'オレンジ色 の Tシャツ を 着ている 人 が 立って います',
]
#prediceted は一つだけじゃないとダメ
predicted={}
predicted['262148']=['人 が オレンジ色 の シャツ を 着て 立って います']

#keyは数字でも文字列でもどっちでもいいけど、ground truth と predicedで対応が取れるように！

#compute blue
scorer=Bleu(4)
score, scores = scorer.compute_score(ground_truth, predicted)
print(scores)
for i,value in enumerate(scores):
    print(i, np.mean(value))  # not same. Blue does not use standard mean.some weighted geometric mean?

#meter requires other thesaurus to 
    
#compute Rouge
scorer=Rouge()
score, scores = scorer.compute_score(ground_truth, predicted)
print(score)
print(np.mean(scores))

#compute CIDEr
scorer=Cider()
score, scores = scorer.compute_score(ground_truth, predicted)
print(score)
print(np.mean(scores))
