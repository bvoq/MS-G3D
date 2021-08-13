import cv2
import mediapipe as mp
import os
import shutil

import json
from google.protobuf.json_format import MessageToDict
from google.protobuf.json_format import MessageToJson

from itertools import product


mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

def stringprod(a, b):
    return [x[0]+x[1] for x in list(product(a,b)) ]

#topics = ['health','kep','police']
##topics = ['kep','police']
#nrs = ['1','2','3','4','5']
#topicsnr = stringprod(topics, nrs)
#signers = ['_signer'+str(i+1) for i in range(7)]
#topicsnrsign = stringprod(topicsnr, signers)
#reps = ['_rep'+str(i+1)+'_glosses' for i in range(5)]
#folders = stringprod(topicsnrsign, reps)

#inpath = '/Users/kdkdk/Documents/mscthesis/Greek_isolated/GSL_isol/'
inpath = './GSL_inbetween/'
#outpath = '/Users/kdkdk/Documents/mscthesis/Greek_mediapipe_holistic_upperbodyver/'
#outpath = './Greek_mediapipe_holistic_better2/'
outpath = './Greek_mediapipe_holistic_inbetween/'

posec = 0
lhandc = 0
rhandc = 0
facec = 0
totc = 0



_, folders, _ = next(os.walk(inpath))
folders.sort()

for f in folders:
    path = inpath+f
    out = outpath+f
    _ ,_, file_list = next(os.walk(inpath+f))
    file_list.sort()
    if not os.path.exists(out):
        os.makedirs(out)

    print(file_list) 
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
      for idx, file in enumerate(file_list):


        print("file: ", f+'/'+file ,"  ", path+'/'+file) 
        image = cv2.imread(path+'/'+file)
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

        #print(final_dict)

        #realidx = handedness_dict['index']
        #label = handedness_dict['label']
        #labels.append(label)

        #final_dict['hand_'+str(idx)] = handedness_dict

        #if results.pose_landmarks:
        #  print(
        #      f'Nose coordinates: ('
        #      f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x * image_width}, '
        #      f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y * image_height})'
        #  )
        # Draw pose, left and right hands, and face landmarks on the image.


        #annotated_image = image.copy()
        #mp_drawing.draw_landmarks(
        #    annotated_image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)
        #mp_drawing.draw_landmarks(
        #    annotated_image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        #mp_drawing.draw_landmarks(
        #    annotated_image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        ## Use mp_holistic.UPPER_BODY_POSE_CONNECTIONS for drawing below when
        ## upper_body_only is set to True.
        #mp_drawing.draw_landmarks(
        #    annotated_image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        #cv2.imwrite(out+file[:-4]+'.png', annotated_image)

        with open(out+'/'+file[:-4]+'.json', 'w') as fout:
            json.dump(final_dict, fout)


    print('pose:',posec,'/',totc,' lhand:',lhandc,'/',totc,' rhands:',rhandc,'/',totc,' face:',facec,'/',totc)
    print('no pose: ', totc-posec, ' no lh: ', totc-lhandc, ' no rh: ', totc-rhandc)








## For webcam input:
#cap = cv2.VideoCapture(0)
#with mp_holistic.Holistic(
#    min_detection_confidence=0.5,
#    min_tracking_confidence=0.5) as holistic:
#  while cap.isOpened():
#    success, image = cap.read()
#    if not success:
#      print("Ignoring empty camera frame.")
#      # If loading a video, use 'break' instead of 'continue'.
#      continue
#
#    # Flip the image horizontally for a later selfie-view display, and convert
#    # the BGR image to RGB.
#    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
#    # To improve performance, optionally mark the image as not writeable to
#    # pass by reference.
#    image.flags.writeable = False
#    results = holistic.process(image)
#
#    # Draw landmark annotation on the image.
#    image.flags.writeable = True
#    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#    mp_drawing.draw_landmarks(
#        image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)
#    mp_drawing.draw_landmarks(
#        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
#    mp_drawing.draw_landmarks(
#        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
#    mp_drawing.draw_landmarks(
#        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
#    cv2.imshow('MediaPipe Holistic', image)
#    if cv2.waitKey(5) & 0xFF == 27:
#      break
#cap.release()
