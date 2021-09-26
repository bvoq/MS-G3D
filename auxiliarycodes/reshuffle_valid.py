import os
import json
import shutil
from google.protobuf.json_format import MessageToDict
from google.protobuf.json_format import MessageToJson
from itertools import product


def stringprod(a, b):
    return [x[0]+x[1] for x in list(product(a,b)) ]

topics = ['health','kep','police']
topics2 = ['extracted_annotations_YGEIA','extracted_annotations_KEP','extracted_annotations_AST']
nrs = ['1','2','3','4','5']
topicsnr = stringprod(topics, nrs)
topicsnr2 = stringprod(topics2, nrs)


#inpath = './Greek_openpose_holistic/'
inpath = './GSL_isol/'
inpath_annotations = './GSL_isol/'
out = './Greek_openpose_hands_merged.csv'
outfolder = './GSL_shuffled/'

merged_json = list()


def repeat(s, count):
    out = ""
    for i in range(count):
       out += s
    return out

tokenequalcheck = dict()
mismatchcount = 0

if not os.path.exists(outfolder):
    os.makedirs(outfolder)
with open(out, 'w', newline='') as fout:
    #json.dump(totaljson, fout)

    for i in range(len(topicsnr)):
        top = topicsnr[i]
        top2 = topicsnr2[i]
    
        aopen = open(inpath_annotations + top2 + '.csv', 'r')
        alines = aopen.readlines()
    
    
        signers = ['_signer'+str(i+1) for i in range(7)]
        topicsnrsign = stringprod([top], signers)
        reps = ['_rep'+str(i+1)+'_glosses' for i in range(5)]
        folders = stringprod(topicsnrsign, reps)
    
        index = 0
        emptylines = 0
        signer = 0
        rep = 0
        for f in folders:
            print(inpath+f)
            _, dirs, _ = next(os.walk(inpath+f))
            dirs.sort()
            glosses_count = len(dirs)
            plusshift = 0 # sometimes one gloss counts as multiple
            prevtoken = "" # this is required for some analysis
            for qi, q in enumerate(dirs):
                path = inpath+f+'/'+q+'/'
                #print(f+'/'+q)
                file_list = os.listdir(path)
                #signer = 34
                #rep = 23
        
                file_list = os.listdir(path)
                file_list.sort()
                tup = alines[index].split("|")
                while len(tup) != 2:
                    index += 1
                    tup = alines[index].split("|")

                if "," == tup[1]:
                    tup[1] = "comma"
                if "," in tup[1]:
                    print("remove comma in ", tup[1])
                    tup[1] = tup[1].replace(",","")

                if " " in tup[1]:
                    tup[1] = tup[1].replace(" ","")

                tup[1] = tup[1].replace("A","Α")
                tup[1] = tup[1].replace("E","Ε")
                tup[1] = tup[1].replace("I","Ι")
                tup[1] = tup[1].replace("K","Κ")
                tup[1] = tup[1].replace("M","Μ")
                tup[1] = tup[1].replace("N","Ν")
                tup[1] = tup[1].replace("O","Ο")
                tup[1] = tup[1].replace("R","Ρ")
                tup[1] = tup[1].replace("S","Σ")
                tup[1] = tup[1].replace("T","Τ")
                tup[1] = tup[1].replace("V","Ω")
                tup[1] = tup[1].replace("X","Χ")
                tup[1] = tup[1].replace("Y","Υ")
                tup[1] = tup[1].replace("Z","Ζ")


                # remove ( annotations
                commaup = ""
                if "(" in tup[1]:
                    for li in range(len(tup[1])-1,-1,-1):
                        if tup[1][li] == '(':
                            tup[1] = tup[1][:li]
                            break




                tup[1] = tup[1].replace("","")

                tup[0] = tup[0].strip()
                tup[1] = tup[1].strip()
                #print(str(i)+"_"+str(qi+plusshift))
                if tup[1] == "":
                    tup[1] = 'empty'

                sentenceandgloss = str(i)+"_"+str(qi+plusshift) # ignores signer id or repetition
                if sentenceandgloss in tokenequalcheck:
                    if tokenequalcheck[ sentenceandgloss ] != tup[1]:
                        if qi+plusshift-1 >= 0 :
                            print("mismatch at ",tup[0], "    prev seq: ", tokenequalcheck[ str(i)+"_"+str(qi+plusshift-1)],",", tokenequalcheck[ sentenceandgloss ], " new: ",prevtoken, ",",tup[1])
                        else:
                            print("mismatch at ",tup[0], "    prev seq: end", tokenequalcheck[ sentenceandgloss ], " new: ",prevtoken, ",",tup[1])
                        mismatchcount += 1

                    if (str(i) + "_"+str(qi+plusshift-1)) in tokenequalcheck and tokenequalcheck[ str(i) + "_"+str(qi+plusshift-1)] == tup[1]:
                        plusshift -= 1
                        print("plusshift -=1", plusshift, " due to ", prevtoken, " followed on ", tup[1])

                    elif tokenequalcheck[ sentenceandgloss] == prevtoken:
                        plusshift += 1
                        print("plusshift +=1", plusshift, " due to ", prevtoken, " followed on ", tup[1])


                else:
                    tokenequalcheck[ sentenceandgloss ] = tup[1]
                    if not os.path.exists(outfolder+tup[1]):
                        os.makedirs(outfolder+tup[1])
                

                prevtoken = tup[1]
                if "+" in tup[1]:
                    pcount = 0
                    for c in tup[1]:
                        if c == "+":
                            plusshift+=1
                    print("plusshift += 1 ", plusshift, " due to ", tup[1], " where ", "+" in tup[1])

                sentenceandgloss = str(i)+"_"+str(qi+plusshift)
                predlabel = tup[1] #tokenequalcheck[sentenceandgloss] if sentenceandgloss in tokenequalcheck else '???'
                predlabelgroup = tokenequalcheck[sentenceandgloss] if sentenceandgloss in tokenequalcheck else '???'
                newfolder = outfolder+predlabel+'/'+f+'_'+q+'_'+predlabelgroup


                shutil.copytree(path, newfolder)

                

                #print(tup[0], " ?= ", f+'/'+q, " with key: ", tup[1])

                #these are not equal due to a naming issue in greeksl where some samples have a signer6.
                #if(tup[0] != f+'/'+q):
                    #print("note, one mistake with 6 probably")
                    #print(tup[0], " => ", f+'/'+q, " with key: ", tup[1])

                #assert(tup[0] == f+'/'+q)
                #collected_dict = dict()
                # note: in the greek dataset some numbers have been fudged unfortunately
                #collected_dict["key"] = f+'/'+q
                #collected_dict["key_orig"] = tup[1]
                #collected_dict["file"] = f+'/'+q



                #for idx, file in enumerate(file_list):
                #  jopen = open(path+file)
                #  jfile = json.load(jopen)
                #  jopen.close()


                #  assert("people" in jfile)
                #  if(1 != len(jfile["people"])):
                #      print("problem: there are ",len(jfile["people"])," people in file: ", f+'/'+q)
                #  assert( 1 == len(jfile["people"]) )

                #  #signer, rep, idx, file, token, 
                #  fout.write(f+"/"+q+","+str(idx)+","+str(signer)+","+tup[1]+",")

                #  # format: x,y,c where c=confidence in [0,1]
                #  assert("pose_keypoints_2d" in jfile["people"][0])
                #  if("pose_keypoints_2d" in jfile["people"][0]):
                #    assert(len(jfile["people"][0]["pose_keypoints_2d"]) == 25*3)
                #    for pi, p in enumerate(jfile["people"][0]["pose_keypoints_2d"]):
                #        fout.write(str(p)+",")

                #  assert("hand_left_keypoints_2d" in jfile["people"][0])
                #  if("hand_left_keypoints_2d" in jfile["people"][0]):
                #      assert(len(jfile["people"][0]["hand_left_keypoints_2d"]) == 21*3)
                #      for pi, p in enumerate(jfile["people"][0]["hand_left_keypoints_2d"]):
                #        fout.write(str(p)+",")


                #  assert("hand_right_keypoints_2d" in jfile["people"][0])
                #  if("hand_right_keypoints_2d" in jfile["people"][0]):
                #      assert(len(jfile["people"][0]["hand_right_keypoints_2d"]) == 21*3)
                #      for pi, p in enumerate(jfile["people"][0]["hand_left_keypoints_2d"]):
                #        fout.write(str(p)+("," if pi < 21*3-1 else ""))


                #  fout.write("\n")
                #  #frames.append(jfile)
                index += 1
                #collected_dict["data"] = frames
                #merged_json.append(collected_dict)
 
            rep = (rep + 1) % 5
            if rep == 0:
                signer = (signer + 1) % 7

   
        print(f, " ", index, "-",emptylines, " ?= ", len(alines))
        assert(index == len(alines))
print("done")
print("mismatches x: ", mismatchcount) # no -=1
