# taken from: https://towardsdatascience.com/entity-embeddings-for-ml-2387eb68e49
import optuna
import pandas as pd
import numpy as np
from scipy import special as spc

sentence = "./forward_bgasclass_ABfiller.txt"
ground_truth = "./groundtruth_ABfiller.txt"

df = pd.read_csv(sentence, sep='\t', header=None)
arr = df.to_numpy()
ground_truth_pd = pd.read_csv(ground_truth, sep='\t', header=None)
ground_truth = ground_truth_pd.to_numpy()

shiftoffset = 15//2
#df.transpose()

backgroundclass = 334
averageThresholding = False  # if false use k < Kappa heuristic, else avg heuristic
removeProblematicGestures = False

# ignore the gestures: ΓΕΝΝΩ (150) and ΕΝΤΑΞΕΙ(34)
if removeProblematicGestures:
    arr = np.log(arr)
    arr[:,34]  = float('-inf')
    arr[:,150] = float('-inf')
    arr=spc.softmax(arr,axis=1)


#[I 2021-09-18 00:45:05,746] Trial 380 finished with value: 10.003399999999942 and parameters: {'threshold': 0.0027337505179572943, 'mingest': 3, 'mingap': 5}. Best is trial 380 with value: 10.003399999999942.



def objective(trial):

    threshold = trial.suggest_uniform("threshold",0.0001,1.0)
    #threshold = 0.070198773686484
    predictions = []
    mingesturesize = 4
    mingapsize = 4
    #mingesturesize = trial.suggest_int("mingesturesize",1,6)
    #mingapsize = trial.suggest_int("mingapsize",1,6)

    #threshold = 0.07032982142757253
    #mingesturesize = 6
    #mingapsize = 2
    print(arr.shape)
    # temporal convolutions over XYZ
    for row in range(arr.shape[0]):
        predictions.append(int(arr[row][backgroundclass] < threshold))
    
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
    
    
    err = 0 # actual error
    totposserr = 0 # total possible error
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
            elif predstate == 2:
                totposserr += 1
            if predstate != -2:
                predstate = -1

        if truth == 0:
            if truthstate == 1:
                err += 1
            elif truthstate == 2:
                totposserr += 1
            if truthstate != -2:
                truthstate = -1

        if pred == 1:
            if predstate == -1:
                err += 1
            elif predstate == -2:
                totposserr += 1
            if predstate != 2:
                predstate = 1

        if truth == 1:
            if truthstate == -1:
                err += 1
            elif truthstate == -2:
                totposserr += 1
            if truthstate != 2:
                truthstate = 1

        if pred == 1 and truth == 1 and predstate == 1 and truthstate == 1:
            predstate = 2
            truthstate = 2

        if pred == 0 and truth == 0 and predstate == -1 and truthstate == -1:
            predstate = -2
            truthstate = -2


        if pred != truth:
            err += 0.00001 # to correct small alignment issues
        #totposserr += 0.00001
        #print("pred/real: ", pred, "/", truth, " ", minindex)

    #for p in predictions:
    #    print(p)
    #with open('outpred_bgasclass.txt','w') as fout:
    #    for p in predictions:
    #        print(p)
    #        print(p, file=fout)

    print("error: ", err, "/", totposserr, "=", (err/totposserr)) # 189
    return err




#objective(0)

study = optuna.create_study(direction="minimize") #optuna.load_study(study_name='study1', storage='sqlite:///study1.db')
study.optimize(objective, n_trials=1000)
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
