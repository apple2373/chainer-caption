#!/usr/bin/env python
# -*- coding: utf-8 -*-

#implementation of LSTM net for caption generation
#inpired from https://github.com/dsanno/chainer-image-caption/blob/master/src/net.py

import chainer
import chainer.functions as F
import chainer.links as L

class ImageCaptionDecorder(chainer.Chain):
    def __init__(self, vocaburary_size, feature_dim=2048, hidden_dim=512,dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        super(ImageCaptionDecorder, self).__init__(
            embed_word=  L.EmbedID(vocaburary_size, hidden_dim),
            embed_image= L.Linear(feature_dim, hidden_dim),
            lstm = L.LSTM(hidden_dim, hidden_dim),
            decode_word = L.Linear(hidden_num, vocaburary_size),
        )

    def input_cnn_feature(self, image_feature, train=True):
        self.lstm.reset_state()
        h = self.embed_image(image_feature)
        self.lstm(F.dropout(h, ratio=self.dropout_ratio, train=train))

    def __call__(self, x, train):
        h = self.embed_word(word)
        h = self.lstm(F.dropout(h, ratio=self.dropout_ratio, train=train))
        h = self.decode_word(F.dropout(h, ratio=self.dropout_ratio, train=train))
