import json
import sys
import os
import shutil

# make sure this code ends with a /
# folder1 = 'gsl_mediapipe/gsl_mediapipe_train/' 
# folder2 = 'ninjagestures/embed/gsl_mediapipe_train/' 

# creating a dataset where both are trained to compare the two datasets.
# label1 = 'gsl_mediapipe/gsl_mediapipe_train_label.json' 
# label2 = 'ninjagestures/embed/gsl_mediapipe_train_label.json'

#label1 = 'gsl_mediapipe/gsl_mediapipe_train_label.json' 
#label2 = 'gsl_mediapipe_inbetween/gsl_mediapipe_train_label.json' 
#
## make sure this code ends with a /
#folder1 = 'gsl_mediapipe/gsl_mediapipe_train/' 
#folder2 = 'gsl_mediapipe_inbetween/gsl_mediapipe_train/'
## folder2 = 'ninjagestures/embed/gsl_mediapipe_train/' 

label1 = 'gsl_mediapipe_woproblematic/gsl_mediapipe_train_label.json' 
label2 = 'gsl_mediapipe_inbetween/gsl_mediapipe_train_label.json' 

# make sure these definitions end with a /
folder1 = 'gsl_mediapipe_woproblematic/gsl_mediapipe_train/' 
folder2 = 'gsl_mediapipe_inbetween/gsl_mediapipe_train/'
outlocation = 'gsl_mediapipe_woproblematicandbg/' 


outfolder = outlocation + 'gsl_mediapipe_train/'
outlabel = outlocation + 'gsl_mediapipe_train_label.json'

if not os.path.exists(outlocation):
    os.makedirs(outlocation)


fopen1 = open(label1)
json1 = json.load(fopen1)

fopen2 = open(label2)
json2 = json.load(fopen2)


json1.update(json2)
for k in json1:
    print(k, " ")

with open(outlabel, 'w', encoding='utf-8') as labelout:
    json.dump(json1, labelout, ensure_ascii=False)

def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)
# copy the json files for merging
shutil.copytree(folder1, outfolder)
for item in os.listdir(folder2):
    shutil.copy2(os.path.join(folder2,item), outfolder)

