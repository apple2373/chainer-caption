#!/bin/bash

cd data
if [ ! -f ResNet50.model ]; then
  wget https://www.dropbox.com/s/eqdmml7kj3545sv/ResNet50.model 
fi
if [ ! -f caption_cn_model40.model ]; then
  wget https://www.dropbox.com/s/680hubm829hjhii/caption_cn_model40.model
fi
if [ ! -f caption_en_model40.model ]; then
  wget https://www.dropbox.com/s/sy4ayrvush0bqle/caption_en_model40.model
fi
if [ ! -f caption_jp_mt_model40.model ]; then
  wget https://www.dropbox.com/s/26dnl6ibpb60kxd/caption_jp_mt_model40.model
fi
if [ ! -f caption_jp_yj_model40.model ]; then
  wget wget https://www.dropbox.com/s/mseh6ls0gvfrnms/caption_jp_yj_model40.model
fi

cd MSCOCO
if [ "$1" == "train" ]; then
  if [ ! -f train2014_ResNet50_features.zip ]; then
    wget  https://www.dropbox.com/s/v3tewruak581uf3/train2014_ResNet50_features.zip
    unzip train2014_ResNet50_features.zip
  fi 
  if [ ! -f val2014_ResNet50_features.zip ]; then
    wget https://www.dropbox.com/s/qi3s55vzgmkiqkh/val2014_ResNet50_features.zip
    unzip val2014_ResNet50_features.zip
  fi 
  if [ ! -f captions_train2014_cn_translation.json ]; then
    wget https://github.com/apple2373/mt-mscoco/raw/master/captions_train2014_cn_translation.json.zip
    unzip captions_train2014_cn_translation.json.zip
    rm captions_train2014_cn_translation.json.zip
  fi 
  if [ ! -f captions_train2014_jp_translation.json.zip ]; then
    wget https://github.com/apple2373/mt-mscoco/raw/master/captions_train2014_jp_translation.json.zip
    unzip captions_train2014_jp_translation.json.zip
    rm captions_train2014_jp_translation.json.zip
  fi 
  if [ ! -f yjcaptions26k_clean.json ]; then
    wget https://github.com/yahoojapan/YJCaptions/raw/master/yjcaptions26k.zip
    unzip yjcaptions26k.zip
    rm yjcaptions26k.zip
  fi 
  if [ ! -f captions_train2014.json ]; then
    wget http://msvocds.blob.core.windows.net/annotations-1-0-3/captions_train-val2014.zip
    unzip captions_train-val2014.zip
    rm captions_train-val2014.zip
  fi 
fi
	
