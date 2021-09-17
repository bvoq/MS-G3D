# taken from: https://towardsdatascience.com/entity-embeddings-for-ml-2387eb68e49
import pandas as pd
import numpy as np

sentence = "./ninjasentenceB1_wbg/forward.csv"
ground_truth = "./groundtruthB1.txt"

df = pd.read_csv(sentence, sep='\t', header=None)
arr = df.to_numpy()
ground_truth_pd = pd.read_csv(ground_truth, sep='\t', header=None)
ground_truth = ground_truth_pd.to_numpy()

shiftoffset = 6
#df.transpose()


# ignore the gestures: ΓΕΝΝΩ (150) and ΕΝΤΑΞΕΙ(34)
#arr[:,150] = 0
#arr[:,34] = 0
predictions = []

threshold = 0.01
for row in range(arr.shape[0]):
    totalsum = 0

    #order = arr[row,:].argsort()
    arr[row,:].sort()
    arr[row,:] = arr[row, ::-1]
    equal_arr = True
    for ia, a in enumerate(arr[row,:]):
        if row > 0 and arr[row-1,ia] != a:
            equal_arr = False
    
    if row > 0 and equal_arr:
        print("done")
        break

    #sorted = np.take(v, order, 0)
    for col in range(arr.shape[1]):
        totalsum += arr[row][col] * arr[row][col]
        #print(arr[row][col])

    newsum = 0
    minindex = -1
    for col in range(arr.shape[1]):
        newsum += arr[row][col] * arr[row][col]

        if newsum >= (1 - threshold) * totalsum:
            minindex = col
            break

    predictions.append(int(minindex <= 9))

mingesturesize = 4
mingapsize = 3
assert(mingapsize <= mingesturesize)

for ip, p in enumerate(predictions):
    if p == 1:
        count = 0
        for back in range(1,mingesturesize+1):
            if ip-back >= 0:
                if predictions[ip-back] == 1:
                    count += 1
                else:
                    break
        for front in range(1,mingesturesize+1):
            if ip+front < len(predictions):
                if predictions[ip+front] == 1:
                    count += 1
                else:
                    break

        if count < mingesturesize:
            predictions[ip] = 0

for ip, p in enumerate(predictions):
    if p == 0:
        count = 0
        for back in range(1,mingesturesize+1):
            if ip-back >= 0:
                if predictions[ip-back] == 0:
                    count += 1
                else:
                    break
        for front in range(1,mingesturesize+1):
            if ip+front < len(predictions):
                if predictions[ip+front] == 0:
                    count += 1
                else:
                    break

        if count < mingapsize:
            predictions[ip] = 1


for row in range(len(predictions)):
    for ia, a in enumerate(arr[row,:]):
        if row > 0 and arr[row-1,ia] != a:
            equal_arr = False
    if row > 0 and equal_arr:
        print("done")
        break

    should_be = 0
    if row+shiftoffset < len(ground_truth):
        should_be = ground_truth[row+6][0]
    print(len(predictions))
    print("pred/real: ", predictions[row], "/", should_be, " ", minindex)

for p in predictions:
    print(p)
