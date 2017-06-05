# -*- coding: utf-8 -*-
#!/usr/bin/env python

import sys
sys.path.append('../coco-caption/')

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import argparse
import json


#parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--true', type=str,help='json file of true captions')
parser.add_argument('--predicted', type=str,help='json file of predicted captions')
parser.add_argument('--output', type=str, help='the filepath of output files')
args = parser.parse_args()

# JSONファイル読み込み
with open(args.predicted,'r') as f:
	results = json.load(f)

#make the input to coco caption evaluator
results_list=[]
for key in results:
	a_caption={}
	image_id= int(key.split(".")[-2].split("_")[-1])
	caption = (" ".join(results[key][0]["sentence"][1:-1])).strip()
	a_caption["image_id"]=image_id
	a_caption["caption"]=caption+"."
	results_list.append(a_caption)

resFile="/tmp/coco_eval.json"
with open(resFile, 'w') as f:
	json.dump(results_list, f, sort_keys=True, indent=4)

# create coco object and cocoRes object
coco = COCO(args.true)
cocoRes = coco.loadRes(resFile)
# create cocoEval object by taking coco and cocoRes
cocoEval = COCOEvalCap(coco, cocoRes)

# evaluate on a subset of images by setting
# please comment out this line when evaluating the full validation set
cocoEval.params['image_id'] = cocoRes.getImgIds()

#evaluate results
cocoEval.evaluate()

# print output evaluation scores
results={}
for metric, score in cocoEval.eval.items():
	results[metric]=score

with open(args.output, 'w') as f:
	json.dump(results, f, sort_keys=True, indent=4)
