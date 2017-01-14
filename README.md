#image caption generation by chainer 

This repository contains an implementation of typical image caption generation based on neural network (i.e. CNN + RNN). The model first extracts the image feature by CNN and then generates captions by RNN. CNN is ResNet50 and RNN is a standard LSTM .

The training data is MSCOCO. I preprocessed MSCOCO images by extracting CNN features in advance. Then I trained the language model to generate captions. Not only English, I trained on Japanese and Chinese. 

I made pre-trained models available. For English captions, the model achieves CIDEr of 0.692 (Otheres are Bleu1: 0.657,  Bleu2: 0.471, Bleu3: 0.327, Bleu4: 0.228,  METEOR: 0.213, ROUGE_L: 0.47)  for the MSCOCO validation dataset. The scores are increased a little bit when the beam search is used. For example, CIDEr is 0.716 with beam size of 5. If you want to achieve a better score, CNN has to be fine-tuned, but I haven’t tried because it’s computationally heavier.  

<img src="sample.png" >

##requirements
chainer 1.19.0  http://chainer.org
and some more packages.  
!!Warning ** Be sure to use chainer 1.19.0. if you want to use for sure**  Chainer is notorious for breaking downward compatibility . If you have another version, no guarantee to work.  
If you are new, I strongly recoomend Anaconda (https://www.continuum.io/downloads) and then install chainer.  
```
#After installing anaconda, you can install chainer, sepcifically, version1.19.0,  in this way. 
pip install chainer==1.19.0 
```

##I just want to generate caption!
OK, first, you need to download the models and other preprocessed files.
```
bash download.sh
```
Then you can generate caption.
```
#English
python sample_code_beam.py \
--rnn-model ./data/caption_en_model40.model \
--cnn-model ./data/ResNet50.model \
--vocab ./data/MSCOCO/mscoco_caption_train2014_processed_dic.json \
--gpu -1 \
--img ./sample_imgs/COCO_val2014_000000185546.jpg \

#Japanese trained from machine translated Japanese (https://github.com/apple2373/mt-mscoco)
python sample_code_beam.py \
--rnn-model ./data/caption_jp_mt_model40.model \
--cnn-model ./data/ResNet50.model \
--vocab ./data/MSCOCO/captions_train2014_jp_translation_processed_dic.json \
--gpu -1 \
--img ./sample_imgs/COCO_val2014_000000185546.jpg \


#Japanese by YJCaptions (https://github.com/yahoojapan/YJCaptions)
python sample_code_beam.py \
--rnn-model ./data/caption_jp_yj_model40.model \
--cnn-model ./data/ResNet50.model \
--vocab ./data/MSCOCO/yjcaptions26k_clean_processed_dic.json \
--gpu -1 \
--img ./sample_imgs/COCO_val2014_000000185546.jpg \

#Chinese trained from machine translated Chinese (https://github.com/apple2373/mt-mscoco)
python sample_code_beam.py \
--rnn-model ./data/caption_cn_model40.model \
--cnn-model ./data/ResNet50.model \
--vocab ./data/MSCOCO/captions_train2014_cn_translation_processed_dic.json \
--gpu -1 \
--img ./sample_imgs/COCO_val2014_000000185546.jpg \

```
See the help for other options. You can, for example, use beam search if you want. 

## I want to run caption generation module as a web API.
I have a simple script for that.
```
cd webapi
python server.py --rnn-model ../data/caption_en_model40.model \
--cnn-model ../data/ResNet50.model \
--vocab ../data/MSCOCO/mscoco_caption_train2014_processed_dic.json \
--gpu -1 \

curl -X POST -F image=@./sample_imgs/COCO_val2014_000000185546.jpg http://localhost:8090/predict
#you should get json
```


##I want to train the model by myself.
* I am trying to update the code so that it can fine-tune CNNs. The interface will be changed.   
I made preprocessed files available. You can download like this.
```
bash download.sh train
```
Then you can train like this.
```
python train_caption_model.py --savedir ./experiment1 --epoch 40 --batch 120 --gpu -1 \
--vocab ./data/MSCOCO/mscoco_caption_train2014_processed_dic.json \
--captions ./data/MSCOCO/MSCOCO/mscoco_caption_train2014_processed.json \
```

##I want to train the model from my own data.
* I am trying to update the code so that it can fine-tune CNNs. The interface will be changed.  s
Alright, you need to do additional amount of work.
```
cd code
#extract features using ResNet50 \
python ResNet_feature_extractor.py --img-dir ../data/MSCOCO/train2014 \
 --out-dir ../data/MSCOCO/train2014_ResNet50_features \
  --gpu -1
```
`--gpu` is GPU id (-1 is CPU).`—img-dir` is the directory that you stores images. `—out-dir` is the directory that the ResNet features will be saved. The file name will be the same, but extension is “.npz”.   
```
#preprocess the json files. you need to have the same structure as MSCOCO json.
python preprocess_MSCOCO_captions.py \
--input ../data/MSCOCO/captions_train2014.json \
--output ../data/MSCOCO/mscoco_caption_train2014_processed.json \
--outdic ../data/MSCOCO/mscoco_caption_train2014_processed_dic.json \
--outfreq ../data/MSCOCO/mscoco_caption_train2014_processed_freq.json \
—-cut 5 \
—-char True 
```
`input` is the json file containing caption. `output` will be the main preprocessed output. `outdic` is the vocabulary file. `outfreq` is the internal file you don’t need it in the training. Just frequency count. `cut` is the cutoff frequency for minor words. Character based chunking will be used when `char` is True. You can use it for non-spaced languages like Japanese and Chinese. 
  
Then you can use my script above for training. 

##I want to fine-tune CNN part. 
Sorry, current implementation does not support it. I am working on it now. Maybe you can read and modify the code. 