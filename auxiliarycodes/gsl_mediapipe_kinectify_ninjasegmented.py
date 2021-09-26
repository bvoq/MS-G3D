import os
import json
from google.protobuf.json_format import MessageToDict
from google.protobuf.json_format import MessageToJson
from itertools import product

### TO MODIFY ###
specialsigner = 6 # between 0-6, this signer will be used for validation
current_labelindex = 334 # the label dedicated to the background

inpath = './Greek_mediapipe_holistic_ninjasentence/'
inpath = './ninjasentenceA4/'
#inpath_annotations = './GSL_shuffled_annotations/'
outlocation = './gsl_mediapipe_ninjasegmentedA4/'
reverselabels = './reverselabels.txt'
outpath_embed = outlocation + 'gsl_mediapipe_embed/'
outlabel_embed = outlocation + 'gsl_mediapipe_embed_label.json'
#outpath_train = outlocation + 'gsl_mediapipe_train/'
#outpath_val = outlocation + 'gsl_mediapipe_val/'
#outlabel_train = outlocation + 'gsl_mediapipe_train_label.json'
#outlabel_val = outlocation + 'gsl_mediapipe_val_label.json'

### END OF MODIFY ###

if not os.path.exists(outpath_embed):
    os.makedirs(outpath_embed)

rdict = dict()
rlabel = open(reverselabels, "r")
rlines = rlabel.readlines()
rlabel.close()
for i in range(len(rlines)):
    rdict[ rlines[i].split(',')[1].strip() ] = i

#if not os.path.exists(outpath_train):
#    os.makedirs(outpath_train)
#
#if not os.path.exists(outpath_val):
#    os.makedirs(outpath_val)


merged_json = list()


def repeat(s, count):
    out = ""
    for i in range(count):
       out += s
    return out


# Format of the parsing:
# {"data": [{"frame_index": 1, "skeleton": [{"pose": [0.630, 0.247, 0.698, 0.291, 0.663, 0.277, 0.626, 0.353, 0.532, 0.380, 0.741, 0.293, 0.661, 0.399, 0.546, 0.421, 0.710, 0.511, 0.712, 0.641, 0.714, 0.764, 0.759, 0.511, 0.728, 0.652, 0.712, 0.775, 0.630, 0.226, 0.642, 0.228, 0.000, 0.000, 0.683, 0.228], "score": [0.975, 0.787, 0.675, 0.821, 0.709, 0.734, 0.852, 0.674, 0.496, 0.531, 0.469, 0.577, 0.530, 0.682, 0.063, 0.917, 0.000, 0.892]},


#{"data": [{"frame_index": 1, "skeleton": [{"pose": [0.630, 0.247, 0.698, 0.291, 0.663, 0.277, 0.626, 0.353, 0.532, 0.380, 0.741, 0.293, 0.661, 0.399, 0.546, 0.421, 0.710, 0.511, 0.712, 0.641, 0.714, 0.764, 0.759, 0.511, 0.728, 0.652, 0.712, 0.775, 0.630, 0.226, 0.642, 0.228, 0.000, 0.000, 0.683, 0.228], "score": [0.975, 0.787, 0.675, 0.821, 0.709, 0.734, 0.852, 0.674, 0.496, 0.531, 0.469, 0.577, 0.530, 0.682, 0.063, 0.917, 0.000, 0.892]}, {"pose": [0.350, 0.187, 0.317, 0.277, 0.272, 0.291, 0.321, 0.380, 0.467, 0.402, 0.362, 0.272, 0.405, 0.353, 0.479, 0.381, 0.278, 0.462, 0.319, 0.611, 0.338, 0.726, 0.338, 0.443, 0.401, 0.546, 0.321, 0.554, 0.336, 0.182, 0.000, 0.000, 0.315, 0.198, 0.000, 0.000], "score": [0.913, 0.849, 0.802, 0.847, 0.435, 0.816, 0.829, 0.741, 0.616, 0.765, 0.720, 0.653, 0.848, 0.150, 0.925, 0.000, 0.752, 0.000]}, {"pose": [0.000, 0.000, 0.016, 0.266, 0.006, 0.272, 0.006, 0.307, 0.000, 0.000, 0.033, 0.258, 0.057, 0.310, 0.045, 0.337, 0.041, 0.334, 0.041, 0.394, 0.061, 0.465, 0.061, 0.326, 0.049, 0.383, 0.102, 0.399, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.006, 0.245], "score": [0.000, 0.530, 0.295, 0.083, 0.000, 0.709, 0.518, 0.365, 0.621, 0.611, 0.746, 0.745, 0.638, 0.664, 0.000, 0.000, 0.000, 0.128]}]}, {"frame_index": 2

totlabels = 0
#labelmap = {} # label -> index
#rlabelmap = [None] * 350 # there are roughly 321 classes

gsl_mediapipe_label_embed = {}
#gsl_mediapipe_label_train = {}
#gsl_mediapipe_label_val = {}

max_frame = 0

_, folders, _ = next(os.walk(inpath))
folders.sort()
for fidx, f in enumerate(folders):
    info = f.split('_')
    #assert(len(info) >= 4) # could be more if token contains _ in name
    signer = int(info[1])
    print("folder: ",f)
    _, _, file_list = next(os.walk(inpath+f))
    file_list.sort()

    # gsl_mediapipe_format
    gsl_mediapipe_format = {}
    gsl_mediapipe_format['data'] = []
    for idx, file in enumerate(file_list):
      if idx+1 > max_frame:
        max_frame = idx+1

      kineticsframe = {}
      kineticsframe['frame_index'] = idx+1
      kineticspose = {}
      kineticspose['pose'] = []
      kineticspose['score'] = []

      jopen = open(inpath+f+'/'+file)
      jfile = json.load(jopen)
      jopen.close()

      if("pose" in jfile):
        assert(len(jfile["pose"]) == 33)
        for pi, p in enumerate(jfile["pose"]):
            kineticspose['pose'].append(str(p["x"]))
            kineticspose['pose'].append(str(p["y"]))
            kineticspose['pose'].append(str(p["z"]))
            kineticspose['score'].append(str(p["visibility"]))
      else:
        print("NO BODY")
        kineticspose['pose' ].extend(["0.0"] * 33*3)
        kineticspose['score'].extend(["0.0"] * 33*1)


      if("left_hand" in jfile):
        assert(len(jfile["left_hand"]) == 21)
        for pi, p in enumerate(jfile["left_hand"]):
            kineticspose['pose'].append(str(p["x"]))
            kineticspose['pose'].append(str(p["y"]))
            kineticspose['pose'].append(str(p["z"]))
            kineticspose['score'].append("1.0")
      else:
        print("NO LH")
        kineticspose['pose'].extend(["0.0"]* 21*3)
        kineticspose['score'].extend(["0.0"]* 21*1)

      if("right_hand" in jfile):
        assert(len(jfile["right_hand"]) == 21)
        for pi, p in enumerate(jfile["right_hand"]):
           kineticspose['pose'].append(str(p["x"]))
           kineticspose['pose'].append(str(p["y"]))
           kineticspose['pose'].append(str(p["z"]))
           kineticspose['score'].append("1.0")
      else:
        print("NO RH")
        kineticspose['pose'].extend(["0.0"]* 21*3)
        kineticspose['score'].extend(["0.0"]* 21*1)

      assert(225 == len(kineticspose['pose']))
      assert(75 == len(kineticspose['score']))
      # ignore face

      kineticsframe['skeleton'] = [kineticspose]
      gsl_mediapipe_format['data'].append(kineticsframe)
      #frames.append(jfile)

    #print("encoded: ", tup[1], " encoded: ", tup[1].encode('utf-8'))
    #encodedstr = str(tup[1].encode('utf-8'))
    #encodedstr = encodedstr.replace('\\x','')
    encodedstr = "background" # creating a background class, tup[1] # keep utf-8
    encodedstr = rlines[fidx].split(',')[1].strip()
    current_labelindex = fidx
    print("fidx: ", fidx, " with ", encodedstr)
    #print("int: ", encodedstr)
    gsl_mediapipe_format['label'] = encodedstr # tup[1]
    gsl_mediapipe_format['label_index'] = current_labelindex

    with open(outpath_embed+f+'.json', 'w', encoding='utf-8') as outgloss:
        json.dump(gsl_mediapipe_format, outgloss, ensure_ascii=False)

    gsl_mediapipe_label_embed[f] = {}
    gsl_mediapipe_label_embed[f]['has_skeleton'] = True
    gsl_mediapipe_label_embed[f]['label'] = encodedstr # tup[1]
    gsl_mediapipe_label_embed[f]['label_index'] = current_labelindex
 
    #if (signer % 7 != specialsigner):
    #    #print("print: ",outpath_train+f+'_'+q+'.json')
    #    #print("signer: ", signer)
    #    with open(outpath_train+f+'.json', 'w', encoding='utf-8') as outgloss:
    #        json.dump(gsl_mediapipe_format, outgloss, ensure_ascii=False)

    #    gsl_mediapipe_label_train[f] = {}
    #    gsl_mediapipe_label_train[f]['has_skeleton'] = True
    #    gsl_mediapipe_label_train[f]['label'] = encodedstr # tup[1]
    #    gsl_mediapipe_label_train[f]['label_index'] = current_labelindex
    #else:
    #    with open(outpath_val+f+'.json', 'w', encoding='utf-8') as outgloss:
    #        json.dump(gsl_mediapipe_format, outgloss, ensure_ascii=False)

    #    gsl_mediapipe_label_val[f] = {}
    #    gsl_mediapipe_label_val[f]['has_skeleton'] = True
    #    gsl_mediapipe_label_val[f]['label'] = encodedstr # tup[1]
    #    gsl_mediapipe_label_val[f]['label_index'] = current_labelindex

    print("very cool idea")

with open(outlabel_embed, 'w', encoding='utf-8') as labelout:
    json.dump(gsl_mediapipe_label_embed, labelout, ensure_ascii=False)
#with open(outlabel_val, 'w', encoding='utf-8') as labelout:
#    #json.dump(gsl_mediapipe_label_val, labelout)
#    json.dump(gsl_mediapipe_label_val, labelout, ensure_ascii=False)

#rlabelout = open("./reverselabels.txt","w", encoding='utf-8')
#for i in range(len(rlabelmap)):
#    if rlabelmap[i] != None:
#        rlabelout.write(str(i)+","+rlabelmap[i]+"\n")
#rlabelout.close()
print("done")
print("max frame size: ", max_frame)
print("total labels: ", totlabels)

