#!/usr/bin/env python
# -*- coding: utf-8 -*-

#implementation of LSTM net for caption generation
#inpired from https://github.com/dsanno/chainer-image-caption/blob/master/src/net.py
#consulted on http://www.monthly-hack.com/entry/2016/10/24/200000
#consulted on http://qiita.com/chantera/items/d8104012c80e3ea96df7

import chainer
import chainer.functions as F
import chainer.links as L

class Image2CaptionDecoder(chainer.Chain):
    def __init__(self, vocaburary_size, img_feature_dim=2048, hidden_dim=512,dropout_ratio=0.5,train=True, n_layers=1):
        super(Image2CaptionDecoder, self).__init__(
            embed_word=  L.EmbedID(vocaburary_size, hidden_dim),
            embed_image= L.Linear(img_feature_dim, hidden_dim),
            lstm = L.NStepLSTM(n_layers=n_layers,in_size=hidden_dim,out_size=hidden_dim,dropout=dropout_ratio),
            decode_word = L.Linear(hidden_dim, vocaburary_size),
        )
        self.dropout_ratio = dropout_ratio
        self.train = train
        self.n_layers=n_layers
        self.hidden_dim=hidden_dim

    def input_cnn_feature(self,hx,cx,image_feature):
        h = self.embed_image(image_feature)
        h = [F.reshape(img_embedding,(1,self.hidden_dim)) for img_embedding in h]#一回　python list/tuple にしないとerrorが出る
        hy, cy, ys  = self.lstm(hx, cx, h, train=self.train)
        return hy,cy

    def __call__(self, hx, cx, caption_batch):
        #hx (~chainer.Variable): Initial hidden states.
        #cx (~chainer.Variable): Initial cell states.
        #xs (list of ~chianer.Variable): List of input sequences.Each element ``xs[i]`` is a :class:`chainer.Variable` holding a sequence.
        xs = [self.embed_word(caption) for caption in caption_batch]
        hy, cy, ys  = self.lstm(hx, cx, xs, train=self.train)
        predicted_caption_batch = [self.decode_word(generated_caption) for generated_caption in ys]
        if self.train:
            loss=0
            for y, t in zip(predicted_caption_batch, caption_batch):
                loss+=F.softmax_cross_entropy(y[0:-1], t[1:])
            return loss/len(predicted_caption_batch)
        else:
            return hy, cy, predicted_caption_batch

class Image2CaptionDecoderOld(chainer.Chain):
    def __init__(self, vocaburary_size, img_feature_dim=2048, hidden_dim=512,dropout_ratio=0.5,train=True):
        self.dropout_ratio = dropout_ratio
        super(Image2CaptionDecoderOld, self).__init__(
            embed_word=  L.EmbedID(vocaburary_size, hidden_dim),
            embed_image= L.Linear(img_feature_dim, hidden_dim),
            lstm = L.LSTM(hidden_dim, hidden_dim),
            decode_word = L.Linear(hidden_dim, vocaburary_size),
        )
        self.train = train

    def input_cnn_feature(self, image_feature):
        self.lstm.reset_state()
        h = self.embed_image(image_feature)
        self.lstm(F.dropout(h, ratio=self.dropout_ratio, train=self.train))

    def __call__(self, cur_word, next_word=None):
        h = self.embed_word(cur_word)
        h = self.lstm(F.dropout(h, ratio=self.dropout_ratio, train=self.train))
        h = self.decode_word(F.dropout(h, ratio=self.dropout_ratio, train=self.train))
        if self.train:
            self.loss = F.softmax_cross_entropy(h, next_word)
            return self.loss
        else:
            return h


if __name__ == '__main__':
    #test code on cpu
    import numpy as np
    image_feature=np.zeros([2,2048],dtype=np.float32)
    x_batch=[[1,2,3,4,2,3,0,2],[1,2,3,3,1]]
    x_batch = [np.array(x,dtype=np.int32) for x in x_batch]
    model=Image2CaptionDecoder(5)
    batch_size=len(x_batch)
    hx=np.zeros((model.n_layers, batch_size, model.hidden_dim), dtype=np.float32)
    cx=np.zeros((model.n_layers, batch_size, model.hidden_dim), dtype=np.float32)
    hx,cx=model.input_cnn_feature(hx,cx,image_feature)
    loss = model(hx, cx, x_batch)

    #test old one
    image_feature=np.zeros([1,2048],dtype=np.float32)
    x_list=np.array([[1,2,3,4,2,3,0,2]],dtype=np.int32).T
    model=Image2CaptionDecoderOld(5)
    model.input_cnn_feature(image_feature)
    loss = 0
    for cur_word, next_word in zip(x_list, x_list[1:]):
        loss += model(cur_word, next_word)

