#!/usr/bin/python
# coding: UTF-8

#output file is [{ "file_path": "path/img.jpg", "captions": ["a caption", "a similar caption" ...] }, ...]


import os
import argparse
import json

import random
random.seed(0)

'''
e.g. 

python3 create_MSCOCO_en_jp_dataset.py \
--en ../data/MSCOCO/captions_train2014.json \
--out-en ../data/MSCOCO/mscoco_en_only_26k.json \
--jp ../data/MSCOCO/yjcaptions26k_clean.json \
--out-jp ../data/MSCOCO/mscoco_jp.json
'''

def read_MSCOCO_json(file_place,args):
        
    f = open(file_place, 'r')
    jsonData = json.load(f)
    f.close()

    imgs = jsonData['images']
    annotations=jsonData['annotations']
    img_id2captions={}
    for caption_data in annotations:
        caption_id=caption_data['id']
        image_id=caption_data['image_id']
        caption=caption_data['caption']
        if image_id not in img_id2captions:
            img_id2captions[image_id]=[caption]
        else:
            img_id2captions[image_id].append(caption)

    # create the json blob
    out = []
    for i,img in enumerate(imgs):
        imgid = img['id']
        jimg = {}
        jimg['file_path'] = os.path.join('train2014', img['file_name'])        
        jimg['captions'] = img_id2captions[imgid]
        jimg['id'] = imgid
        out.append(jimg)
            
    return out

def save_raw_json(captions,outfile,args):
    with open(args.en, 'r') as f:
        jsonData=json.load(f)
    captions={"annotations":captions,"images":jsonData['images'],"type":"captions","info":"dummy, not all image ids show up in captions","licenses":"dummy"}
    with open(outfile, 'w') as f:
        json.dump(captions, f, sort_keys=True, indent=4)    

if __name__ == '__main__':
    #parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--en', type=str,help='json file of MSCOCO formatted captions for English')
    parser.add_argument('--jp', type=str,help='json file of MSCOCO formatted captions for Japanese')
    parser.add_argument('--out-jp', type=str, help='the output file for Japanese')
    parser.add_argument('--out-en', type=str, help='the output file for English')
    args = parser.parse_args()

    #load jsonfile
    en_captions=read_MSCOCO_json(args.en,args)
    jp_captions=read_MSCOCO_json(args.jp,args)

    en_img_ids=[caption["id"] for caption in en_captions]
    jp_img_ids=[caption["id"] for caption in jp_captions]

    common_img_ids = set(en_img_ids) & set(jp_img_ids)
    remain_img_ids = set(en_img_ids) - set(jp_img_ids)

    # common_img_ids=list(common_img_ids)
    # random.shuffle(common_img_ids)

    print("common image ids",len(common_img_ids))

    new_english_dataset=[]
    for caption in en_captions:
        img_id=caption["id"]
        if img_id in common_img_ids:
            new_english_dataset.append(caption)

    new_japanese_dataset=[]
    for caption in jp_captions:
        img_id=caption["id"]
        if img_id in common_img_ids:
            new_japanese_dataset.append(caption)

    assert(len(new_english_dataset) == len(new_japanese_dataset))
    print(args.out_jp)
    print(args.out_en)

    with open(args.out_en,'w') as f:
        json.dump(new_english_dataset,f, sort_keys=True, indent=4,ensure_ascii=False)
    with open(args.out_jp,'w') as f:
        json.dump(new_japanese_dataset,f, sort_keys=True, indent=4,ensure_ascii=False)