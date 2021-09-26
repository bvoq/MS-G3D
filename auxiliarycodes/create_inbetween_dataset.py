import os
import json
import shutil
from google.protobuf.json_format import MessageToDict
from google.protobuf.json_format import MessageToJson
from itertools import product
import cv2


in_isol = './GSL_isol/'
in_annotations = './GSL_shuffled_annotations/'
in_continuous = './GSL_continuous/'
out_inbetween = './GSL_inbetween/'
splitinfo = 'split_info.json'

mininbetweenframes = 4


def stringprod(a, b):
    return [x[0]+x[1] for x in list(product(a,b)) ]

def equalImage(original, duplicate):
    if original.shape == duplicate.shape:
        difference = cv2.subtract(original, duplicate)
        b, g, r = cv2.split(difference)
        return cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0


#topics = ['health','kep','police']
##topics = ['kep','police']
#nrs = ['1','2','3','4','5']
#topicsnr = stringprod(topics, nrs)
#signers = ['_signer'+str(i+1) for i in range(7)]
#topicsnrsign = stringprod(topicsnr, signers)
#reps = ['_rep'+str(i+1)+'_glosses' for i in range(5)]
#folders = stringprod(topicsnrsign, reps)

topics = ['health','kep','police']
topics2 = ['extracted_annotations_YGEIA','extracted_annotations_KEP','extracted_annotations_AST']
nrs = ['1','2','3','4','5']
topicsnr = stringprod(topics, nrs)
topicsnr2 = stringprod(topics2, nrs)


totalsentencecount = 0
goodsentencecount = 0
totaltokencount = 0
goodtokencount = 0
averageLen = 0.0
#inpath = '/Users/kdkdk/Documents/mscthesis/Greek_isolated/GSL_isol/'
#outpath = '/Users/kdkdk/Documents/mscthesis/Greek_mediapipe_holistic_upperbodyver/'

_, contdirs, _ = next(os.walk(in_continuous))
contdirs.sort()
contdirs_cindex = -1 #+ 175 + 172 # resp. offsets if you want to start kep or police
contsent_cindex = -1 
contimg_cindex = -1

contdirs_cindex += 1
_, contsentences, _ = next(os.walk(in_continuous + contdirs[contdirs_cindex]))
contsentences.sort()
#contsentences.sort(key=lambda x: int("".join(list(filter(str.isdigit, x)))) )
print(contsentences)
contsent_cindex = -1

contsent_cindex += 1
totalsentencecount += 1
_, _, contimages = next(os.walk(in_continuous + contdirs[contdirs_cindex] + '/' + contsentences[contsent_cindex]))

contimages.sort()
#contimages.sort(key=lambda x: int("".join(list(filter(str.isdigit, x)))) )
contimg_cindex = -1

contimg_cindex += 1
cimgpath = in_continuous + contdirs[contdirs_cindex] + '/' + contsentences[contsent_cindex]+'/'+contimages[contimg_cindex]
cimg = cv2.imread(cimgpath)

sentencewithnotokenproblems = True
tokenlist = []
tokenstartpos = []
tokenendpos = []
listoftokensentences = []
listoftokenlist = []
listoftokenstartpos = []
listoftokenendpos = []

#imagesmissing = ["health1_signer1_rep3_glosses/glosses0018/frame_0018.jpg","health1_signer2_rep5_glosses/glosses0007/frame_0020.jpg",
#        "health1_signer3_rep2_glosses/glosses0027/frame_0020.jpg","health1_signer3_rep3_glosses/glosses0050/frame_0019.jpg", "health1_signer3_rep4_glosses/glosses0035/frame_0012.jpg",
#        "health1_signer3_rep4_glosses/glosses0052/frame_0022.jpg","health1_signer3_rep5_glosses/glosses0026/frame_0019.jpg", "health1_signer3_rep5_glosses/glosses0041/frame_0000.jpg",
#        "health1_signer4_rep1_glosses/glosses0051/frame_0000.jpg","health1_signer4_rep3_glosses/glosses0035/frame_0010.jpg", "health1_signer5_rep1_glosses/glosses0035/frame_0013.jpg",
#        "health1_signer5_rep3_glosses/glosses0040/frame_0018.jpg","health1_signer6_rep1_glosses/glosses0040/frame_0013.jpg", "health1_signer6_rep3_glosses/glosses0040/frame_0015.jpg"]
#        
imagesmissing = ["health2_signer1_rep4_glosses/glosses0005/frame_0001.jpg", "health2_signer1_rep4_glosses/glosses0005/frame_0002.jpg", "health2_signer1_rep4_glosses/glosses0005/frame_0003.jpg", "kep1_signer7_rep2_glosses/glosses0089/frame_0002.jpg","kep1_signer7_rep2_glosses/glosses0089/frame_0003", "kep2_signer2_rep1_glosses/glosses0075/frame_0014.jpg"]

#h2s1r4g5 = stringprod(["health2_signer1_rep4_glosses/glosses0005/frame_00","health2_signer1_rep4_glosses/glosses0006/frame_00", "health4_signer3_rep5_glosses/glosses0060/frame_00", "health4_signer3_rep5_glosses/glosses0074/frame_00", "health4_signer3_rep5_glosses/glosses0080/frame_00", "health4_signer5_rep5_glosses/glosses0027/frame_00", "kep1_signer1_rep4_glosses/glosses0064/frame_00", "kep1_signer7_rep2_glosses/glosses0089/frame_00", "kep1_signer4_rep2_glosses/glosses0045/frame_00","kep1_signer4_rep2_glosses/glosses0046/frame_00", "kep1_signer4_rep2_glosses/glosses0047/frame_00","kep1_signer7_rep2_glosses/glosses0090/frame_00", "kep2_signer1_rep3_glosses/glosses0080/frame_00","kep2_signer1_rep3_glosses/glosses0081/frame_00", "kep2_signer2_rep1_glosses/glosses0075/frame_00","kep2_signer3_rep1_glosses/glosses0011/frame_00","kep2_signer3_rep3_glosses/glosses0022/frame_00","kep2_signer3_rep3_glosses/glosses0037/frame_00","kep2_signer4_rep4_glosses/glosses0000/frame_00","kep2_signer4_rep4_glosses/glosses0004/frame_00","kep2_signer4_rep4_glosses/glosses0010/frame_00","kep2_signer4_rep4_glosses/glosses0037/frame_00", "kep2_signer4_rep4_glosses/glosses0042/frame_00","kep2_signer4_rep4_glosses/glosses0044/frame_00","kep2_signer4_rep4_glosses/glosses0049/frame_00","kep2_signer4_rep4_glosses/glosses0050/frame_00"]#"kep2_signer4_rep4_glosses/glosses0075/frame_00", "kep2_signer4_rep4_glosses/glosses0076/frame_00", "kep2_signer5_rep1_glosses/glosses0000/frame_00","kep2_signer5_rep1_glosses/glosses0001/frame_00"], map(lambda x : str(x).zfill(2), list(range(1,50))) )
h2s1r4g5 = stringprod(["health2_signer1_rep4_glosses/glosses0005/frame_00","health2_signer1_rep4_glosses/glosses0006/frame_00", "health4_signer3_rep5_glosses/glosses0060/frame_00", "health4_signer3_rep5_glosses/glosses0074/frame_00", "health4_signer3_rep5_glosses/glosses0080/frame_00", "health4_signer5_rep5_glosses/glosses0027/frame_00", "kep1_signer1_rep4_glosses/glosses0064/frame_00", "kep1_signer7_rep2_glosses/glosses0089/frame_00", "kep1_signer4_rep2_glosses/glosses0045/frame_00","kep1_signer4_rep2_glosses/glosses0046/frame_00", "kep1_signer4_rep2_glosses/glosses0047/frame_00","kep1_signer7_rep2_glosses/glosses0090/frame_00", "kep2_signer1_rep3_glosses/glosses0080/frame_00","kep2_signer1_rep3_glosses/glosses0081/frame_00", "kep2_signer2_rep1_glosses/glosses0075/frame_00","kep2_signer3_rep1_glosses/glosses0011/frame_00","kep2_signer3_rep3_glosses/glosses0022/frame_00","kep2_signer3_rep3_glosses/glosses0037/frame_00","kep2_signer4_rep4_glosses/glosses0000/frame_00","kep2_signer4_rep4_glosses/glosses0004/frame_00","kep2_signer4_rep4_glosses/glosses0010/frame_00","kep2_signer4_rep4_glosses/glosses0037/frame_00", "kep2_signer4_rep4_glosses/glosses0042/frame_00","kep2_signer4_rep4_glosses/glosses0044/frame_00","kep2_signer4_rep4_glosses/glosses0049/frame_00","kep2_signer4_rep4_glosses/glosses0050/frame_00", "kep2_signer4_rep4_glosses/glosses0075/frame_00","kep2_signer4_rep4_glosses/glosses0076/frame_00","kep2_signer5_rep1_glosses/glosses0075/frame_00","kep2_signer5_rep2_glosses/glosses0075/frame_00","kep3_signer3_rep4_glosses/glosses0061/frame_00","kep3_signer3_rep4_glosses/glosses0062/frame_00","police2_signer2_rep2_glosses/glosses0004/frame_00","police2_signer2_rep5_glosses/glosses0073/frame_00","police3_signer3_rep2_glosses/glosses0062/frame_00","police3_signer3_rep2_glosses/glosses0063/frame_00","police4_signer1_rep3_glosses/glosses0067/frame_00","police4_signer1_rep3_glosses/glosses0068/frame_00","police4_signer1_rep3_glosses/glosses0069/frame_00", "police4_signer4_rep3_glosses/glosses0067/frame_00", "police4_signer4_rep3_glosses/glosses0068/frame_00", "police4_signer4_rep3_glosses/glosses0069/frame_00", "police5_signer5_rep3_glosses/glosses0060/frame_00", "police5_signer5_rep3_glosses/glosses0061/frame_00","police5_signer5_rep3_glosses/glosses0062/frame_00", "police5_signer7_rep1_glosses/glosses0060/frame_00", "police5_signer7_rep1_glosses/glosses0061/frame_00", "police5_signer7_rep1_glosses/glosses0062/frame_00"], map(lambda x : str(x).zfill(2), list(range(1,50))) )

# kep1_signer4_rep2_glosses/glosses0045/frame_00
h2s1r4g5 = stringprod(h2s1r4g5, [".jpg"])
imagesmissing = imagesmissing + h2s1r4g5
imagesmissing = stringprod([in_isol], imagesmissing)

sentencesmissing = ["health2_signer1_rep4_sentences/sentences0001/frame_0008.jpg"]
sentencesmissing = stringprod([in_isol], sentencesmissing)

prevtoken = "???" # ??? if prev token is invalid or beginning of sentence.
inbetweencounter = 0

for i, top in enumerate(topicsnr):
    #print(top)
    top2 = topicsnr2[i]

    aopen = open(in_annotations + top2 + '.csv', 'r')
    alines = aopen.readlines()

    signers = ['_signer'+str(i+1) for i in range(7)]
    topicsnrsign = stringprod([top], signers)
    reps = ['_rep'+str(i+1)+'_glosses' for i in range(5)]
    folders = stringprod(topicsnrsign, reps)

    index = 0
    emptylines = 0
    signer = ""
    rep = 0

    for f in folders:
        print("folder: ", f)
        signer = f.split('_')[1]
        signer = signer[-1:] # only pick the last
        _, dirs, _ = next(os.walk(in_isol+f))
        dirs.sort()
        glosses_count = len(dirs)
        imseq = []
        for qi, q in enumerate(dirs):
            path = in_isol+f+'/'+q+'/'
            totaltokencount += 1
            file_list = os.listdir(path)
    
            #if not os.path.exists(out):
            #    os.makedirs(out)
            file_list = os.listdir(path)
            file_list.sort()
    
            tup = alines[index].split("|")
            while len(tup) != 2:
                index += 1
                tup = alines[index].split("|")

            tup[0] = tup[0].strip()
            tup[1] = tup[1].strip()

            current_labelindex = 0
            #if tup[1] in labelmap:
            #    current_labelindex = labelmap[tup[1]]

            #else:
            #    current_labelindex = totlabels
            #    labelmap[tup[1]] = totlabels
            #    rlabelmap[totlabels] = tup[1]
            #    totlabels += 1

            if qi == 0:
                prevtoken = "???"

            if tup[1] == '???':# or qi == 0: # there is no prev. gloss to gloss 0
                #print("skipped ??? ", tup[0])
                prevtoken = "???"
                index += 1
                imseq = []
                sentencewithnotokenproblems = False
                continue

            #print(tup[0], " ?= ", f+'/'+q, " with key: ", tup[1])
            if(tup[0] != f+'/'+q):
                print("note, one mistake with 6 probably")
                print(tup[0], " => ", f+'/'+q, " with key: ", tup[1])
            #assert(tup[0] == f+'/'+q)
            averageLen += len(file_list)

            for idx, file in enumerate(file_list):
                #print("file: ", f+'/'+q+'/'+file) 
                #if idx == 0 or idx+1 == len(file_list) or path+file in imagesmissing:

                # often times the last or first are missing so i just skip them.
                if idx == 0 or idx+1 == len(file_list):
                    continue

                if path+file in imagesmissing:
                    #print("truth set")
                    prevtoken = "???"
                    sentencewithnotokenproblems = False
                    continue

                image = cv2.imread(path+file)
                image_height, image_width, _ = image.shape

                
                #if path+file in imagesmissing:
                # 1. keep pushing cimg
                # 2. make sure to create new gesture if prevtokenqqq = True
                miss_cimg = cimg
                miss_contimg_cindex = contimg_cindex
                miss_cimgpath = cimgpath
                miss_contsent_cindex = contsent_cindex
                miss_contdirs_cindex = contdirs_cindex


                imseq = []
                misscounter = 0

                while (not equalImage(image, cimg)):
                    # missing images from sign language dataset
                    #print("tada ", path+file, " ", cimgpath)

                    if prevtoken != "???":
                        #print("mc: ", misscounter)
                        imseq.append(cimgpath)

                    misscounter += 1
                    #print("tada ", path+file, " ", cimgpath)
                    if(misscounter > 1000):
                        print("tada ", path+file, " ", cimgpath)
                        assert(False)
                        exit(0)

                    if contimg_cindex+1 >= len(contimages):
                        if(sentencewithnotokenproblems):
                            listoftokensentences.append( in_continuous + contdirs[contdirs_cindex] + '/' + contsentences[contsent_cindex] )
                            listoftokenlist.append( tokenlist )
                            listoftokenstartpos.append( tokenstartpos )
                            listoftokenendpos.append( tokenendpos )
                            print("good sentence: ",in_continuous + contdirs[contdirs_cindex] + '/' + contsentences[contsent_cindex])
                            print("tokenlist: ", tokenlist)
                            print("tokenstartpos: ", tokenstartpos)
                            print("tokenendpos: ", tokenendpos)
                            goodsentencecount += 1
                            goodtokencount += len(tokenlist)
                            # export the entire sentence in an N^2 fashion. might end up being too big so for now only do health?


                        if contsent_cindex+1 >= len(contsentences):
                            assert(contdirs_cindex+1 < len(contdirs))
                            contdirs_cindex += 1
                            if os.path.exists(in_continuous + contdirs[contdirs_cindex]):
                                _, contsentences, _ = next(os.walk(in_continuous + contdirs[contdirs_cindex]))
                                contsentences.sort()
                                contsent_cindex = -1
                            else:
                                assert(False)
                        
                        sentencewithnotokenproblems = True
                        tokenlist = []
                        tokenstartpos = []
                        tokenendpos = []

                        contsent_cindex += 1
                        totalsentencecount += 1
                        _, _, contimages = next(os.walk(in_continuous + contdirs[contdirs_cindex] + '/' + contsentences[contsent_cindex]))
                        contimages.sort()
                        contimg_cindex = -1
                        prevtoken = "???"# in between sentences
                        imseq=[]
                        #print("sset true")

                    contimg_cindex += 1
                    cimgpath = in_continuous + contdirs[contdirs_cindex] + '/' + contsentences[contsent_cindex]+'/'+contimages[contimg_cindex]
                    cimg = cv2.imread(cimgpath)

                if idx == 1:
                    tokenlist.append( tup[1] )
                    print(tup[1])
                    assert( (not " " in tup[1]) and (not "," in tup[1]))
                    tokenstartpos.append( contimg_cindex ) 

                if idx+2 == len(file_list):
                    tokenendpos.append( contimg_cindex )

                if prevtoken != "???" and len(imseq) >= mininbetweenframes:
                    fpname = out_inbetween+str(inbetweencounter).zfill(5)+"_"+signer+"_"+prevtoken+"_"+tup[1] # TODO ADD TWO TOKEN NAMES HERE
                    inbetweencounter += 1
                    if not os.path.exists(fpname):
                        os.makedirs(fpname)

                    for iim, im in enumerate(imseq):
                        print('copy ', im, ' to ', fpname+'/'+str(iim).zfill(3)+'.jpg')
                        shutil.copyfile(im, fpname+'/'+str(iim).zfill(3)+'.jpg')


                #print("match ", path+file, " ", cimgpath)

                #if contimg_cindex+1 >= len(contimages):
                #    if contsent_cindex+1 >= len(contsentences):
                #        assert(contdirs_cindex+1 < len(contdirs))
                #        contdirs_cindex += 1
                #        if os.path.exists(in_continuous + contdirs[contdirs_cindex]):
                #            _, contsentences, _ = next(os.walk(in_continuous + contdirs[contdirs_cindex]))
                #            contsentences.sort()
                #            contsent_cindex = -1
                #        else:
                #            assert(False)
                #    
                #    contsent_cindex += 1
                #    _, _, contimages = next(os.walk(in_continuous + contdirs[contdirs_cindex] + '/' + contsentences[contsent_cindex]))
                #    contimages.sort()
                #    contimg_cindex = -1
                #    prevtokenqqq = True # in between sentences

                #contimg_cindex += 1
                #cimgpath = in_continuous + contdirs[contdirs_cindex] + '/' + contsentences[contsent_cindex]+'/'+contimages[contimg_cindex]
                #cimg = cv2.imread(cimgpath)

                print("goodsentencecount: ", goodsentencecount, "/", totalsentencecount)
                print("goodtokencount: ", goodtokencount, "/", totaltokencount)
                print("averageLen ", (averageLen / totaltokencount) )



            prevtoken = tup[1]
            imseq = []
            index += 1 # index for annotations

        prevtoken = "???" 
    assert(index == len(alines))

final_dict = {}
final_dict['sentences'] = listoftokensentences
final_dict['tokens'] = listoftokenlist
final_dict['startpos'] = listoftokenstartpos
final_dict['endpos'] = listoftokenendpos

print("exporting json: ", splitinfo)

with open(splitinfo, 'w', encoding='utf-8') as fout:
    json.dump(final_dict, fout, ensure_ascii=False)

print("done")
print("goodsentencecount: ", goodsentencecount, "/", totalsentencecount)
print("goodtokencount: ", goodtokencount, "/", totaltokencount)
print("averageLen ", (averageLen / totaltokencount) )
