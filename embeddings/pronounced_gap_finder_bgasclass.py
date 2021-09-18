# taken from: https://towardsdatascience.com/entity-embeddings-for-ml-2387eb68e49
import optuna
import pandas as pd
import numpy as np
from scipy import special as spc

sentence = "./forward_ABfiller.csv"
ground_truth = "./groundtruth_ABfiller.txt"

df = pd.read_csv(sentence, sep='\t', header=None)
arr = df.to_numpy()
ground_truth_pd = pd.read_csv(ground_truth, sep='\t', header=None)
ground_truth = ground_truth_pd.to_numpy()

shiftoffset = 15//2
#df.transpose()


averageThresholding = False  # if false use k < Kappa heuristic, else avg heuristic

# ignore the gestures: ΓΕΝΝΩ (150) and ΕΝΤΑΞΕΙ(34)
#arr = np.log(arr)
#arr[:,204] = float('-inf')
#arr[:,34] = float('-inf')
#arr[:,150] = float('-inf')
#arr=spc.softmax(arr,axis=1)
#arr2 = arr.copy()
arr = np.log(arr)
arr[:,34]  = float('-inf')
arr[:,150] = float('-inf')
arr=spc.softmax(arr,axis=1)


#[I 2021-09-18 00:45:05,746] Trial 380 finished with value: 10.003399999999942 and parameters: {'threshold': 0.0027337505179572943, 'mingest': 3, 'mingap': 5}. Best is trial 380 with value: 10.003399999999942.



def objective(trial):

    #threshold = 0.02
    #threshold = 0.005

    threshold = trial.suggest_uniform("threshold",0.0001,0.01)
    predictions = []
    print(arr.shape)
    # temporal convolutions over XYZ
    for row in range(arr.shape[0]):
        totalsum = 0
    
        #order = arr[row,:].argsort()
        arr[row,:].sort()
        arr[row,:] = arr[row, ::-1]
        equal_arr = True
        for ia, a in enumerate(arr[row,:]):
            if row > 0 and arr[row-1,ia] != a:
                equal_arr = False
        #if row > 0 and equal_arr:
        #    print("done")
        #    break
        #sorted = np.take(v, order, 0)
        for col in range(arr.shape[1]):
            totalsum += arr[row][col] * arr[row][col]

        newsum = 0
        avg = 0
        minindex = -1
        for col in range(arr.shape[1]):
            newsum += arr[row][col] * arr[row][col]
            avg += arr[row][col]
            if newsum >= (1 - threshold) * totalsum:
                minindex = col
                break

        avg = avg / (minindex+1) 

        if averageThresholding:
            predictions.append(int(avg >= avgb))
        else:
            predictions.append(int(minindex <= kgap))
    
    #assert(mingapsize <= mingesturesize)
    #print("predlen ", len(predictions)) 
    
    for noisesize in range(1,max(mingesturesize, mingapsize)+1):
        if noisesize <= mingesturesize:
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
            
                    if count < noisesize:
                        predictions[ip] = 0
    
        if noisesize <= mingapsize:
            for ip, p in enumerate(predictions):
                if p == 0:
                    count = 0
                    for back in range(1,mingapsize+1):
                        if ip-back >= 0:
                            if predictions[ip-back] == 0:
                                count += 1
                            else:
                                break
                    for front in range(1,mingapsize+1):
                        if ip+front < len(predictions):
                            if predictions[ip+front] == 0:
                                count += 1
                            else:
                                break
            
                    if count < noisesize:
                        predictions[ip] = 1
    
    
    err = 0
    predstate = 0
    truthstate = 0
    for row in range(len(predictions)):

        pred = predictions[row]
        truth = 0
        if row+shiftoffset<len(ground_truth):
            truth = ground_truth[row+shiftoffset]

        if pred == 0:
            if predstate == 1:
                err += 1
            if predstate != -2:
                predstate = -1

        if truth == 0:
            if truthstate == 1:
                err += 1
            if truthstate != -2:
                truthstate = -1

        if pred == 1:
            if predstate == -1:
                err += 1
            if predstate != 2:
                predstate = 1

        if truth == 1:
            if truthstate == -1:
                err += 1
            if truthstate != 2:
                truthstate = 1

        if pred == 1 and truth == 1 and predstate == 1 and truthstate == 1:
            predstate = 2
            truthstate = 2

        if pred == 0 and truth == 0 and predstate == -1 and truthstate == -1:
            predstate = -2
            truthstate = -2


        if pred != truth:
            err += 0.00001 # small complaints about alignment
        #print("pred/real: ", pred, "/", truth, " ", minindex)

    for p in predictions:
        print(p)

    with open('outpred.txt','w') as fout:
        for p in predictions:
            print(p)
            print(p, file=fout)

    return err




#objective(0)

study = optuna.create_study(direction="minimize") #optuna.load_study(study_name='study1', storage='sqlite:///study1.db')
study.optimize(objective, n_trials=1)
print('Best value: {} (params: {})\n'.format(study.best_value, study.best_params))

#for p in predictions:
#    print(p)
#
#with open('outpred.txt','w') as fout:
#    for p in predictions:
#        print(p)
#        print(p, file=fout)
#
#print(arr[0,:])
