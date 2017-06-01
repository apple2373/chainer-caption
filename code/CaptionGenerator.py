# -*- coding: utf-8 -*-
#!/usr/bin/env python

'''
caption generation module by beam search
currently you cannot use efficient generation by batch. Batch size is always one 

beam search might have a small bug or not the most efficient, but seems to be ok.
'''

import os
import chainer 

import numpy as np
import json
import math
from copy import deepcopy

from chainer import cuda
import chainer.functions as F
from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers
from chainer import serializers

from image_loader import Image_loader
from ResNet50 import ResNet
from Image2CaptionDecoder import Image2CaptionDecoder

#priority queue
#reference: http://www.bogotobogo.com/python/python_PriorityQueue_heapq_Data_Structure.php
try:
    import Queue as Q  # ver. < 3.0
except ImportError:
    import queue as Q

import heapq

class CaptionGenerator(object):
    def __init__(self,rnn_model_place,cnn_model_place,dictonary_place,beamsize=3,depth_limit=50,gpu_id=-1,first_word="<sos>",hidden_dim=512,mean="imagenet"):
        self.gpu_id=gpu_id
        self.beamsize=beamsize
        self.depth_limit=depth_limit
        self.image_loader=Image_loader(mean)
        self.index2token=self.parse_dic(dictonary_place)

        self.cnn_model=ResNet()
        serializers.load_hdf5(cnn_model_place, self.cnn_model)
        self.cnn_model.train = False

        self.rnn_model=Image2CaptionDecoder(len(self.token2index),hidden_dim=hidden_dim)
        if len(rnn_model_place) > 0:
            serializers.load_hdf5(rnn_model_place, self.rnn_model)
        self.rnn_model.train = False

        self.first_word=first_word

        #Gpu Setting
        global xp
        if self.gpu_id >= 0:
            xp = cuda.cupy 
            cuda.get_device(gpu_id).use()
            self.cnn_model.to_gpu()
            self.rnn_model.to_gpu()
        else:
            xp=np

    def parse_dic(self,dictonary_place):
        with open(dictonary_place, 'r') as f:
            json_file = json.load(f)
        if len(json_file) < 10:#this is ad-hock. I need to distinguish new format and old format...
            self.token2index = { word['word']:word['idx'] for word in json_file["words"]}
        else:
            self.token2index = json_file

        return {v:k for k,v in self.token2index.items()}

    def successor(self,current_state):
        '''
        Args:
            current_state: a stete, python tuple (hx,cx,path,cost)
                hidden: hidden states of LSTM
                cell: cell states LSTM
                path: word indicies so far as a python list  e.g. initial is self.token2index["<sos>"]
                cost: negative log likelihood

        Returns:
            k_best_next_states: a python list whose length is the beam size. possible_sentences[i] = {"indicies": list of word indices,"cost":negative log likelihood so far}

        '''

        word=[xp.array([current_state["path"][-1]],dtype=xp.int32)]
        hx=current_state["hidden"]
        cx=current_state["cell"]
        hy, cy, next_words=self.rnn_model(hx,cx,word)

        word_dist=F.softmax(next_words[0]).data[0]#possible next word distributions
        k_best_next_sentences=[]
        for i in range(self.beamsize):
            next_word_idx=int(xp.argmax(word_dist))
            k_best_next_sentences.append(\
                {\
                "hidden":hy,\
                "cell":cy,\
                "path":deepcopy(current_state["path"])+[next_word_idx],\
                "cost":current_state["cost"]-xp.log(word_dist[next_word_idx])
                }\
                )
            word_dist[next_word_idx]=0

        return hy, cy, k_best_next_sentences

    def beam_search(self,initial_state):
        '''
        Beam search is a graph search algorithm! So I use graph search abstraction

        Args:
            initial state: an initial stete, python tuple (hx,cx,path,cost)
            each state has 
                hx: hidden states
                cx: cell states
                path: word indicies so far as a python list  e.g. initial is self.token2index["<sos>"]
                cost: negative log likelihood

        Returns:
            captions sorted by the cost (i.e. negative log llikelihood)
        '''
        found_paths=[]
        top_k_states=[initial_state]
        while (len(found_paths) < self.beamsize):
            #forward one step for all top k states, then only select top k after that
            new_top_k_states=[]
            for state in top_k_states:
                #examine to next five possible states
                hy, cy, k_best_next_states = self.successor(state)
                for next_state in k_best_next_states:
                    new_top_k_states.append(next_state)
            selected_top_k_states=heapq.nsmallest(self.beamsize, new_top_k_states, key=lambda x : x["cost"])

            #within the selected states, let's check if it is terminal or not.
            top_k_states=[]
            for state in selected_top_k_states:
                #is goal state? -> yes, then end the search
                if state["path"][-1] == self.token2index["<eos>"] or len(state["path"])==self.depth_limit:
                    found_paths.append(state)
                else:
                    top_k_states.append(state)

        return sorted(found_paths, key=lambda x: x["cost"]) 

    def beam_search0(self,initial_state):
        #original one. This takes much memory
        '''
        Beam search is a graph search algorithm! So I use graph search abstraction
        Args:
            initial state: an initial stete, python tuple (hx,cx,path,cost)
            each state has 
                hx: hidden states
                cx: cell states
                path: word indicies so far as a python list  e.g. initial is self.token2index["<sos>"]
                cost: negative log likelihood
        Returns:
            captions sorted by the cost (i.e. negative log llikelihood)
        '''
        found_paths=[]
        q = Q.PriorityQueue()
        q.put((0,initial_state))
        while (len(found_paths) < self.beamsize):
            i=0
            # this is just a one step ahead? 
            while not q.empty():
                if i == self.beamsize:
                    break
                state=q.get()[1]
                #is goal state? -> yes, then end the search
                if state["path"][-1] == self.token2index["<eos>"] or len(state["path"])==self.depth_limit:
                    found_paths.append(state)
                    continue
                #examine to next five possible states and add to priority queue 
                hy, cy, k_best_next_states = self.successor(state)
                for next_state in k_best_next_states:
                    q.put((state["cost"],next_state))
                i+=1

        return sorted(found_paths, key=lambda x: x["cost"]) 

    def generate(self,image_file_path):
        '''
        Args:
            image_file_path: image_file_path
        '''
        img=self.image_loader.load(image_file_path)
        return self.generate_from_img(img)


    def generate_from_img_feature(self,image_feature):
        if self.gpu_id >= 0:
            image_feature=cuda.to_gpu(image_feature)

        batch_size=1
        hx=xp.zeros((self.rnn_model.n_layers, batch_size, self.rnn_model.hidden_dim), dtype=xp.float32)
        cx=xp.zeros((self.rnn_model.n_layers, batch_size, self.rnn_model.hidden_dim), dtype=xp.float32)
        
        hy,cy = self.rnn_model.input_cnn_feature(hx,cx,image_feature)


        initial_state={\
                    "hidden":hy,\
                    "cell":cy,\
                    "path":[self.token2index[self.first_word]],\
                    "cost":0,\
                }\

        captions=self.beam_search(initial_state)

        caption_candidates=[]
        for caption in captions:
            sentence= [self.index2token[word_idx] for word_idx in caption["path"]]
            log_likelihood = -float(caption["cost"])#cost is the negative log likelihood
            caption_candidates.append({"sentence":sentence,"log_likelihood":log_likelihood})

        return caption_candidates

    def generate_from_img(self,image_array):
        '''Generate Caption for an Numpy Image array
        
        Args:
            image_array: numpy array of image

        Returns:
            list of generated captions, sorted by the cost (i.e. negative log llikelihood)

            The structure is [caption,caption,caption,...]
            Where caption = {"sentence": a generated sentence as a python list of word, "log_likelihood": The log llikelihood of the generated sentence} 

        '''
        if self.gpu_id >= 0:
            image_array=cuda.to_gpu(image_array)
        image_feature=self.cnn_model(image_array, "feature").data.reshape(1,1,2048)#次元が一つ多いのは、NstepLSTMはsequaenceとみなすから。(sequence size, batch size, feature dim)ということ

        return self.generate_from_img_feature(image_feature)

if __name__ == '__main__':
    xp=np
    #test code on cpu
    caption_generator=CaptionGenerator(
        rnn_model_place="../experiment1/caption_model1.model",\
        cnn_model_place="../data/ResNet50.model",\
        dictonary_place="../data/MSCOCO/mscoco_caption_train2014_processed_dic.json",\
        beamsize=3,depth_limit=50,gpu_id=-1,)

    # batch_size=1
    # hx=xp.zeros((caption_generator.rnn_model.n_layers, batch_size, caption_generator.rnn_model.hidden_dim), dtype=xp.float32)
    # cx=xp.zeros((caption_generator.rnn_model.n_layers, batch_size, caption_generator.rnn_model.hidden_dim), dtype=xp.float32)
    # img=caption_generator.image_loader.load("../sample_imgs/COCO_val2014_000000185546.jpg")
    # image_feature=caption_generator.cnn_model(img, "feature").data.reshape(1,1,2048)#次元が一つ多いのは、NstepLSTMはsequaenceとみなすから。(sequence size, batch size, feature dim)ということ
    
    # hy,cy = caption_generator.rnn_model.input_cnn_feature(hx,cx,image_feature)
    # initial_state={\
    #             "hidden":hy,\
    #             "cell":cy,\
    #             "path":[caption_generator.token2index["<sos>"]],\
    #             "cost":0,\
    #             }\

    # #successor test
    # next_states= caption_generator.successor(initial_state)
    # print next_states

    # print "beam search test"
    # captions= caption_generator.beam_search(initial_state)
    # print captions

    captions = caption_generator.generate("../sample_imgs/COCO_val2014_000000185546.jpg")
    for caption in captions:
        print(caption["sentence"])
        print(caption["log_likelihood"])

