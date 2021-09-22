# taken from: https://towardsdatascience.com/entity-embeddings-for-ml-2387eb68e49
import sys
import pandas as pd
import numpy as np
from random import *
from libsvm.svmutil import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score

#dataset = "embeddings_tottrained"
dataset = "embeddings_bgasclass_tot"
dataset = "embeddings_bgnoised_tot"
dataset = "embeddings_standard_tot"

dataset = sys.argv[1] 
embedding = sys.argv[2]
print("loss: ", dataset, " with embedding ", embedding)

method = "forward_nosoftmax"
df = pd.read_csv('./'+dataset+'/'+method+'_all.csv', sep='\t', header=None)
method2 = "embeddings"
df2 = pd.read_csv('./'+dataset+'/'+method2+'_all.csv', sep='\t', header=None)

start = 5628-1
end = 5818-1

y, unusedx = svm_read_problem('./'+dataset+'/'+method+'_libsvm.data')

X1 = df.values
X2 = df2.values

# for testing embedding on a single vertex

#X = X1
if embedding == "A":
	X = X2
elif embedding == "B":
	X = X1
elif embedding == "AB":
	X = np.concatenate((X1,X2),axis=1)
else:
	assert(False)


#X = list(X)

print(type(X))
print(type(y))
y = np.around(y)

print(X[end-3:])
print(y[0:100])
print(y[end-20:])
print(len(X), " vs. ", len(y))
#X = df[['Xval', 'Xval_t1', 'Xval_t2', 'Xval_t3', 'Xval_t4', 'Xval_t5']].values
#y = df['Ytval'].values



include_val = False
def oneShotBagging(method, baggingcount, Xtrain, ytrain, Xtest, ytest):

    #bagging = [[]] * (end-start) # can you see why this doesn't work ^^
    bagging=[[] for i in range(start,end) ]
    #def objective(trial):
    #trial.suggest
    totcount = 0
    ccount = 0
    ccount_label = dict()
    totcount_label = dict()
    for q in range(0,baggingcount):
        # NeighborsClassifier(3),
        #SVC(kernel="linear", C=0.025),
        #SVC(gamma=2, C=1),
        #GaussianProcessClassifier(1.0 * RBF(1.0)),
        #DecisionTreeClassifier(max_depth=5),
        #RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        #MLPClassifier(alpha=1, max_iter=1000),
        #AdaBoostClassifier(),
        #GaussianNB(),
        #QuadraticDiscriminantAnalysis()]
        # default values
        reg = MLPClassifier(hidden_layer_sizes=[200])

        if method == "SVC":
            reg = SVC()
        elif method == "GaussianProcessClassifier":
            reg = GaussianProcessClassifier()
        elif method == "DecisionTreeClassifier":
            reg = DecisionTreeClassifier()
        elif method == "RandomForestClassifier":
            reg = RandomForestClassifier()
        elif method == "MLPClassifier":
            reg = MLPClassifier()
        elif method == "AdaBoostClassifier":
            reg = AdaBoostClassifier()
        elif method == "GaussianNB":
            reg = GaussianNB()
        elif method == "QuadraticDiscriminantAnalysis":
            reg = QuadraticDiscriminantAnalysis()
        elif method == "KNeighborsClassifier":
            reg = KNeighborsClassifier(n_neighbors=1)
        else:
            print("unknown method: ", method)
            assert(False)

        reg.fit(Xtrain, ytrain)
        ypred = reg.predict(Xtest)
        assert(len(ypred) == len(Xtest) and len(ypred) == len(ytest))
        for j in range(len(Xtest)):
            bagging[j].append(round(ypred[j]))

    totcount = 0
    ccount = 0
    ccount_label = dict()
    totcount_label = dict()
    for j in range(len(Xtest)):
        ypred = max(set(bagging[j]), key=bagging[j].count)
        la = int(ytest[j])
        if ypred == la:
            if not la in ccount_label:
                ccount_label[la] = 0
            ccount_label[la] += 1
            ccount += 1

        totcount += 1
        if not la in totcount_label:
            totcount_label[la] = 0
        totcount_label[la] += 1

    #print(f"{method} bagged {baggingcount} accuracy: {ccount}/{totcount}={ccount/totcount:.4f}")
    #for k in ccount_label.keys():
    #    print(f"{k}: {ccount_label[k]}/{totcount_label[k]}={ccount_label[k]/totcount_label[k]:.4f}")
    return ccount


def getAndPrintResultsAllToAll(method, baggingcount):

    #bagging = [[]] * (end-start) # can you see why this doesn't work ^^
    bagging=[[] for i in range(start,end) ]
    #def objective(trial):
    #trial.suggest
    totcount = 0
    ccount = 0
    ccount_label = dict()
    totcount_label = dict()
    for q in range(0,baggingcount):
        for li in range(start, end):
            # NeighborsClassifier(3),
            #SVC(kernel="linear", C=0.025),
            #SVC(gamma=2, C=1),
            #GaussianProcessClassifier(1.0 * RBF(1.0)),
            #DecisionTreeClassifier(max_depth=5),
            #RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
            #MLPClassifier(alpha=1, max_iter=1000),
            #AdaBoostClassifier(),
            #GaussianNB(),
            #QuadraticDiscriminantAnalysis()]
            # default values
            reg = MLPClassifier(hidden_layer_sizes=[200])

            if method == "SVC":
                reg = SVC()
            elif method == "GaussianProcessClassifier":
                reg = GaussianProcessClassifier()
            elif method == "DecisionTreeClassifier":
                reg = DecisionTreeClassifier()
            elif method == "RandomForestClassifier":
                reg = RandomForestClassifier()
            elif method == "MLPClassifier":
                reg = MLPClassifier()
            elif method == "AdaBoostClassifier":
                reg = AdaBoostClassifier()
            elif method == "GaussianNB":
                reg = GaussianNB()
            elif method == "QuadraticDiscriminantAnalysis":
                reg = QuadraticDiscriminantAnalysis()
            elif method == "KNeighborsClassifier1":
                reg = KNeighborsClassifier(n_neighbors=1)
            elif method == "KNeighborsClassifier2":
                reg = KNeighborsClassifier(n_neighbors=2)
            elif method == "KNeighborsClassifier3":
                reg = KNeighborsClassifier(n_neighbors=3)
            elif method == "KNeighborsClassifier4":
                reg = KNeighborsClassifier(n_neighbors=4)
            elif method == "KNeighborsClassifier5":
                reg = KNeighborsClassifier(n_neighbors=5)
            elif method == "KNeighborsClassifier6":
                reg = KNeighborsClassifier(n_neighbors=6)
            elif method == "KNeighborsClassifier7":
                reg = KNeighborsClassifier(n_neighbors=7)
            elif method == "KNeighborsClassifier8":
                reg = KNeighborsClassifier(n_neighbors=8)
            elif method == "KNeighborsClassifier9":
                reg = KNeighborsClassifier(n_neighbors=9)
            elif method == "KNeighborsClassifier10":
                reg = KNeighborsClassifier(n_neighbors=10)
            elif method == "KNeighborsClassifier11":
                reg = KNeighborsClassifier(n_neighbors=11)
            elif method == "KNeighborsClassifier12":
                reg = KNeighborsClassifier(n_neighbors=12)
            else:
                print("unknowl method: ", method)
                assert(False)


            if li == end-1:
                if include_val:
                    reg.fit(X[:li],y[:li])
                else:
                    reg.fit(X[start:li],y[start:li])
            elif li == start:
                reg.fit(X[li+1:end], y[li+1:end])
            else:
                #if include_val:
                #    reg.fit(np.concatenate((X[:li],X[(li+1):])),y[:li]+y[(li+1):])
                #else:
                Xcon = np.concatenate((X[start:li],X[(li+1):]), axis=0)
                ycon = np.concatenate((y[start:li],y[(li+1):]), axis=0)
                reg.fit(Xcon, ycon)


            ypred = reg.predict([X[li]])
            la = int(y[li])
            bagging[li-start].append(round(ypred[0]))

    totcount = 0
    ccount = 0
    ccount_label = dict()
    totcount_label = dict()
    for li in range(start,end):
        ypred = max(set(bagging[li-start]), key = bagging[li-start].count)
        la = int(y[li])
        if ypred == la:
            if not la in ccount_label:
                ccount_label[la] = 0
            ccount_label[la] += 1
            ccount += 1

        totcount += 1
        if not la in totcount_label:
            totcount_label[la] = 0
        totcount_label[la] += 1


    print(f"{method} bagged {baggingcount} accuracy: {ccount}/{totcount}={ccount/totcount:.4f}")
    for k in ccount_label.keys():
        print(f"{k}: {ccount_label[k]}/{totcount_label[k]}={ccount_label[k]/totcount_label[k]:.4f}")


# one-shot bagging
# create the dataset

ntrainperclass = 4
baggingcount = 10

cor = np.zeros(13)
tot = np.zeros(13)

cstart = 320#0
cend = 332+1
model = "MLPClassifier"
for ya in range(320,332+1):
    for attempts in range(1000):
        if attempts % 1 == 0:
            print("attempts: ", attempts)
        Xtrain = np.zeros( ((cend-cstart)*ntrainperclass,X.shape[1]) )
        ytrain = np.zeros( ((cend-cstart)*ntrainperclass) )
        Xtest = np.zeros( (1, X.shape[1]) )
        ytest = np.zeros( (1) )
        r = randint(start, end)
        while int(y[r]) != ya:
            r = randint(start, end)
        prevpicked = dict()
        prevpicked[r]=True
        Xtest[0,:] = X[r,:]
        ytest[0] = y[r]
        for i in range(cstart,cend):
            for k in range(ntrainperclass):
                q = randint(0,end)
                while int(y[q]) != i or q in prevpicked:
                    q = randint(0,end)

                prevpicked[q] = True
                Xtrain[(i-cstart)*ntrainperclass+k,:] = X[q,:]
                ytrain[(i-cstart)*ntrainperclass+k] = i

        cor[int(y[r])-320] += oneShotBagging(model,baggingcount,Xtrain,ytrain,Xtest,ytest)
        tot[int(y[r])-320] += len(ytest)

ctot = 0
ttot = 0
for i in range(320,332+1):
    ctot += int(cor[i-320])
    ttot += int(tot[i-320])
    print(i,"&\t"+ str(int(cor[i-320])).strip()+"/"+str(int(tot[i-320])).strip()+ "="+str(cor[i-320]/tot[i-320]))

print("Total &\t"+ str(ctot).strip()+"/"+str(ttot).strip()+ "="+str(ctot/ttot))
print("loss: ", dataset, " with embedding ", embedding, " model ", model)

#getAndPrintResultsAllToAll("KNeighborsClassifier1", baggingcount)
#getAndPrintResultsAllToAll("KNeighborsClassifier2", baggingcount)
#getAndPrintResultsAllToAll("KNeighborsClassifier3", baggingcount)
#getAndPrintResultsAllToAll("KNeighborsClassifier4", baggingcount)
#getAndPrintResultsAllToAll("KNeighborsClassifier5", baggingcount)
#getAndPrintResultsAllToAll("KNeighborsClassifier6", baggingcount)
#getAndPrintResultsAllToAll("KNeighborsClassifier7", baggingcount)
#getAndPrintResultsAllToAll("KNeighborsClassifier8", baggingcount)
#getAndPrintResultsAllToAll("KNeighborsClassifier9", baggingcount)
#getAndPrintResultsAllToAll("KNeighborsClassifier10", baggingcount)
#getAndPrintResultsAllToAll("KNeighborsClassifier11", baggingcount)
#getAndPrintResultsAllToAll("KNeighborsClassifier12", baggingcount)
#getAndPrintResultsAllToAll("SVC", baggingcount)
##getAndPrintResultsAllToAll("GaussianProcessClassifier", baggingcount)
##getAndPrintResultsAllToAll("DecisionTreeClassifier", baggingcount)
#getAndPrintResultsAllToAll("RandomForestClassifier", baggingcount)
#getAndPrintResultsAllToAll("MLPClassifier", baggingcount)
##getAndPrintResultsAllToAll("AdaBoostClassifier", baggingcount)
#getAndPrintResultsAllToAll("GaussianNB", baggingcount)
##getAndPrintResultsAllToAll("QuadraticDiscriminantAnalysis", baggingcount)
