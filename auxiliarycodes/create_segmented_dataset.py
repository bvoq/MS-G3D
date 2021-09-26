import cv2
import mediapipe as mp
import os
import json
import shutil
import pandas as pd
from google.protobuf.json_format import MessageToDict
from google.protobuf.json_format import MessageToJson
from itertools import product

from google.protobuf.json_format import MessageToDict
from google.protobuf.json_format import MessageToJson

from itertools import product


mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic


splitinfo = 'split_info.json' # generated from create_inbetween_dataset.py
out_segmented = './Greek_mediapipe_holistic_segmented/'
reverselabels = './reverselabels.txt'
noofsentences = 297+1
segmentwidth = 10
specialsigner = 6 # only take segmented instances of the cross subject.


tokenstolabel = dict()
tokenstolabel = []

rdict = dict()
rlabel = open(reverselabels, "r")
rlines = rlabel.readlines()
rlabel.close()
for i in range(len(rlines)):
    rdict[ rlines[i].split(',')[1].strip() ] = i



posec = 0
lhandc = 0
rhandc = 0
facec = 0
totc = 0

with open(splitinfo, 'r') as fin:
    data = json.load(fin)
    print(len(data['sentences']))
    

    for i in range(0, min( noofsentences, len(data['sentences']) )):
        assert( len(data['startpos']) == len(data['endpos']) and len(data['startpos']) == len(data['tokens']) )

        sentence = data['sentences'][i]
        signer = int( sentence.split('/')[-2].split('_')[1][-1] )
        print(i, ": ", signer)
        if signer != specialsigner:
            continue
        file_list = os.listdir(data['sentences'][i])
        file_list.sort()

        if len(file_list) < segmentwidth:
            print("skipped due to sentence being too small.")
            continue

        print("detecting sentence: ", data['sentences'][i])
        print("tokens: ", data['tokens'][i])
        print("tokensid: ", list(map(lambda x : rdict[x], data['tokens'][i])))
        print("startpos: ", data['startpos'][i])
        print("endpos: ", data['endpos'][i])
        for j in range(0,len(file_list) - segmentwidth):
            outfolder = out_segmented+'sent_'+str(signer)+'_'+str(i).zfill(3)+'_'+str(j).zfill(3)+'/'
            if not os.path.exists(outfolder):
                os.makedirs(outfolder)

            with mp_holistic.Holistic(
                static_image_mode=True,#assumed that every image is different if True!!
                model_complexity=2,
                # note older version has upper body only:
                # upper_body_only=True,
                # where: upper_body_only=True,#default=1 see https://solutions.mediapipe.dev/holistic#upper_body_only 25 vs 33 track points
                smooth_landmarks=True,#default ignored if staticimagemode=true
                min_detection_confidence=0.0,#default=0.5
                min_tracking_confidence=0.0, #default=0.5
                ) as holistic:

                for k in range(0, segmentwidth):
                    im = data['sentences'][i] +'/'+ file_list[j+k]
                    #print("copying ", im, " to ", outfolder+str(k).zfill(3)+'.jpg')
                    #shutil.copyfile(im,outfolder+str(k).zfill(3)+'.jpg')

                    image = cv2.imread(im)
                    image_height, image_width, _ = image.shape
                    # Convert the BGR image to RGB before processing.
                    results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                    final_dict = dict()
                    if results.pose_landmarks:
                        final_dict['pose'] = MessageToDict(results.pose_landmarks)['landmark'] #.landmark
                        assert(len(final_dict['pose']) == 33)
                        #print("size pose: ", len(final_dict['pose']))
                        posec += 1
                    else:
                        print("NO POSE!")
                    if results.left_hand_landmarks:
                        final_dict['left_hand'] = MessageToDict(results.left_hand_landmarks)['landmark']
                        assert(len(final_dict['left_hand']) == 21)
                        #print("size lh: ", len(final_dict['left_hand']))
                        lhandc += 1
                    else:
                        print("NO LH")
                        handlsubstitute = [dict()] * 21
                        handlsubstitute[0] = final_dict['pose'][15]
                        handlsubstitute[1] = final_dict['pose'][15]
                        handlsubstitute[2] = final_dict['pose'][21]
                        handlsubstitute[3] = final_dict['pose'][21]
                        handlsubstitute[4] = final_dict['pose'][21]
                        for ik in range(5,13):
                            handlsubstitute[ik] = final_dict["pose"][19]
                        for ik in range(13,21):
                            handlsubstitute[ik] = final_dict["pose"][17]
                        final_dict['left_hand'] = handlsubstitute


                    if results.right_hand_landmarks:
                        final_dict['right_hand'] = MessageToDict(results.right_hand_landmarks)['landmark']
                        assert(len(final_dict['right_hand']) == 21)
                        #print("size rh: ", len(final_dict['right_hand']))
                        rhandc += 1
                    else:
                        print("NO RH")
                        handrsubstitute = [dict()] * 21
                        handrsubstitute[0] = final_dict['pose'][16]
                        handrsubstitute[1] = final_dict['pose'][16]
                        handrsubstitute[2] = final_dict['pose'][22]
                        handrsubstitute[3] = final_dict['pose'][22]
                        handrsubstitute[4] = final_dict['pose'][22]
                        for ik in range(5,13):
                            handrsubstitute[ik] = final_dict["pose"][20]
                        for ik in range(13,21):
                            handrsubstitute[ik] = final_dict["pose"][18]
                        final_dict['right_hand'] = handrsubstitute

                    if results.face_landmarks:
                        #safe some space by not including the face:
                        #final_dict['face'] = MessageToDict(results.face_landmarks)['landmark']
                        #assert(len(final_dict['face']) == 468)
                        facec += 1

                    totc += 1


                    with open(outfolder+(str(k).zfill(3))+'.json', 'w') as fout:
                        json.dump(final_dict, fout)

