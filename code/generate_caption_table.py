#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
generate caption table
'''
import sys
import json
import os
import argparse
import numpy as np
import math
from image_loader import Image_loader
import shutil

class HTMLPrinter(object):
    def __init__(self,file_path):
        self.file=open(file_path, 'w')

    def reset(self,file_path):
        self.file=open(file_path, 'w')

    def write(self,line):
        self.file.write(str(line.encode('utf_8') )+"\n")

    def close(self):
        self.file.close()
        self.file=None

#Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dir',default='../data/AD/AD_viziometrics_figures/', type=str,help='path to the image directory')
parser.add_argument('--predicted',default='../experiment_ad_tl/AD_visiometrics_figure_titles_val_predicted_beam3.json', type=str,help='path to the predicted json file')
parser.add_argument('--output',default="../output1", type=str,help='output directory name')
args = parser.parse_args()

if not os.path.isdir(args.output):
    os.makedirs(args.output)
    print("made the save directory", args.output)
image_dir=args.output+"/images/"
if not os.path.isdir(image_dir):
    os.makedirs(image_dir)

image_loader=Image_loader()

with open(args.predicted, 'r') as f:
    predictions=json.load(f)

html=HTMLPrinter(file_path=args.output+"/captions.html")

i=0
html.write('<html><body><table border="1">')

for image_filename,caption in predictions.items():
    # if image_filename[-4:]=='.npz':
    #     image_filename=image_filename[0:-4].split("/")[-1]
    sys.stdout.write("\r%d" % i)
    sys.stdout.flush()
    i+=1
    image_file_path=args.dir+image_filename+".jpg"
    image_np=image_loader.load(image_file_path,expand_batch_dim=False)
    cropped_image_file=image_dir+image_filename+"_cropped.jpg"
    image_loader.save(image_np,cropped_image_file)
    shutil.copy2(image_file_path, image_dir)

    html.write("<tr>")
    html.write("<td>%d</td>"%i)
    html.write("<td>")
    data_uri = open(cropped_image_file, 'rb').read().encode('base64').replace('\n', '')
    img_tag = '<img src="data:image/png;base64,{0}">'.format(data_uri)
    #html.write('<img src="./imgaes/%s" alt="%s" height="224" width="224">'%(image_filename+"_cropped.jpg",image_filename))
    html.write(img_tag)
    html.write("</td>")
    html.write("<td>")
    # html.write("true caption:")
    # html.write(caption["true"])
    # html.write('<br /><br />')
    for predicted_cap in caption["captions"]:
    # for predicted_cap in caption:
        html.write(str(predicted_cap["log_likelihood"])+'<br />')
        html.write( (" ".join(predicted_cap["sentence"]).replace("<","&lt;").replace(">","&gt;")) +'<br />')
    html.write("</td>")
    html.write("</tr>")

html.write("</table></body></html>")
html.close()