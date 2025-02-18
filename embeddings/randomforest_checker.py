# taken from: https://towardsdatascience.com/entity-embeddings-for-ml-2387eb68e49
import pandas as pd
import numpy as np
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

dataset = "embeddings_bgasclass_tot"
dataset = "embeddings_tottrained"
dataset = "embeddings_bgnoised_tot"
dataset = "embeddings_standard_tot"

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
#X = X2
X = np.concatenate((X1,X2),axis=1)


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


print(type(y))


totcount = 0
ccount = 0
ccount_label = dict()
totcount_label = dict()
for li in range(start, end):
    #KNeighborsClassifier(3),
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
    reg = SVC()
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
    if round(ypred[0]) == la:
        if not la in ccount_label:
            ccount_label[la] = 0
        ccount_label[la] += 1
        ccount += 1

    totcount += 1
    if not la in totcount_label:
        totcount_label[la] = 0
    totcount_label[la] += 1

print(f"SVC accuracy: {ccount}/{totcount}={ccount/totcount:.4f}")
for k in ccount_label.keys():
    print(f"{k}: {ccount_label[k]}/{totcount_label[k]}={ccount_label[k]/totcount_label[k]:.4f}")



totcount = 0
ccount = 0
ccount_label = dict()
totcount_label = dict()
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
    reg = GaussianProcessClassifier()
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
    if round(ypred[0]) == la:
        if not la in ccount_label:
            ccount_label[la] = 0
        ccount_label[la] += 1
        ccount += 1

    totcount += 1
    if not la in totcount_label:
        totcount_label[la] = 0
    totcount_label[la] += 1

print(f"GaussianProcessClassifier accuracy: {ccount}/{totcount}={ccount/totcount:.4f}")
for k in ccount_label.keys():
    print(f"{k}: {ccount_label[k]}/{totcount_label[k]}={ccount_label[k]/totcount_label[k]:.4f}")



totcount = 0
ccount = 0
ccount_label = dict()
totcount_label = dict()
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
    reg = QuadraticDiscriminantAnalysis()
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
    if round(ypred[0]) == la:
        if not la in ccount_label:
            ccount_label[la] = 0
        ccount_label[la] += 1
        ccount += 1

    totcount += 1
    if not la in totcount_label:
        totcount_label[la] = 0
    totcount_label[la] += 1

print(f"QuadraticDiscriminantAnalysis accuracy: {ccount}/{totcount}={ccount/totcount:.4f}")
for k in ccount_label.keys():
    print(f"{k}: {ccount_label[k]}/{totcount_label[k]}={ccount_label[k]/totcount_label[k]:.4f}")


totcount = 0
ccount = 0
ccount_label = dict()
totcount_label = dict()
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
    reg = GaussianNB()
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
    if round(ypred[0]) == la:
        if not la in ccount_label:
            ccount_label[la] = 0
        ccount_label[la] += 1
        ccount += 1

    totcount += 1
    if not la in totcount_label:
        totcount_label[la] = 0
    totcount_label[la] += 1

print(f"GaussianNB accuracy: {ccount}/{totcount}={ccount/totcount:.4f}")
for k in ccount_label.keys():
    print(f"{k}: {ccount_label[k]}/{totcount_label[k]}={ccount_label[k]/totcount_label[k]:.4f}")



totcount = 0
ccount = 0
ccount_label = dict()
totcount_label = dict()
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
    reg = MLPClassifier()
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
    if round(ypred[0]) == la:
        if not la in ccount_label:
            ccount_label[la] = 0
        ccount_label[la] += 1
        ccount += 1

    totcount += 1
    if not la in totcount_label:
        totcount_label[la] = 0
    totcount_label[la] += 1

print(f"MLPClassifier accuracy: {ccount}/{totcount}={ccount/totcount:.4f}")
for k in ccount_label.keys():
    print(f"{k}: {ccount_label[k]}/{totcount_label[k]}={ccount_label[k]/totcount_label[k]:.4f}")



totcount = 0
ccount = 0
ccount_label = dict()
totcount_label = dict()
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
    reg = AdaBoostClassifier()
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
    if round(ypred[0]) == la:
        if not la in ccount_label:
            ccount_label[la] = 0
        ccount_label[la] += 1
        ccount += 1

    totcount += 1
    if not la in totcount_label:
        totcount_label[la] = 0
    totcount_label[la] += 1

print(f"AdaBoostClassifier accuracy: {ccount}/{totcount}={ccount/totcount:.4f}")
for k in ccount_label.keys():
    print(f"{k}: {ccount_label[k]}/{totcount_label[k]}={ccount_label[k]/totcount_label[k]:.4f}")



totcount = 0
ccount = 0
ccount_label = dict()
totcount_label = dict()
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
    reg = DecisionTreeClassifier()
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
    if round(ypred[0]) == la:
        if not la in ccount_label:
            ccount_label[la] = 0
        ccount_label[la] += 1
        ccount += 1

    totcount += 1
    if not la in totcount_label:
        totcount_label[la] = 0
    totcount_label[la] += 1

print(f"Decision tree accuracy: {ccount}/{totcount}={ccount/totcount:.4f}")
for k in ccount_label.keys():
    print(f"{k}: {ccount_label[k]}/{totcount_label[k]}={ccount_label[k]/totcount_label[k]:.4f}")




totcount = 0
ccount = 0
ccount_label = dict()
totcount_label = dict()
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
    reg = RandomForestClassifier(n_estimators=100, max_depth=None)
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
    #print(la, " ?= ",round(ypred[0]))
    if round(ypred[0]) == la:
        if not la in ccount_label:
            ccount_label[la] = 0
        ccount_label[la] += 1
        ccount += 1

    totcount += 1
    if not la in totcount_label:
        totcount_label[la] = 0
    totcount_label[la] += 1

print(f"Random forest accuracy: {ccount}/{totcount}={ccount/totcount:.4f}")
for k in ccount_label.keys():
    print(f"{k}: {ccount_label[k]}/{totcount_label[k]}={ccount_label[k]/totcount_label[k]:.4f}")






for kn in range(1,10):
    totcount = 0
    ccount = 0
    ccount_label = dict()
    totcount_label = dict()
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
        reg = KNeighborsClassifier(n_neighbors=kn)
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
        if round(ypred[0]) == la:
            if not la in ccount_label:
                ccount_label[la] = 0
            ccount_label[la] += 1
            ccount += 1
    
        totcount += 1
        if not la in totcount_label:
            totcount_label[la] = 0
        totcount_label[la] += 1

    print(f"KNN ({str(kn)}) accuracy: {ccount}/{totcount}={ccount/totcount:.4f}")
    for k in ccount_label.keys():
        print(f"{k}: {ccount_label[k]}/{totcount_label[k]}={ccount_label[k]/totcount_label[k]:.4f}")
    








#
#emb_xs =1 #embrows,emd
#emb_valid_xs = 1
#rf = RandomForestClassifier(n_estimators=40, max_samples=100000,
#                            max_features=.5, min_samples_leaf=5)
#rf = rf.fit(X, y)
#roc_auc_score(rf.predict(emb_valid_xs), to.valid.y)

#m = RandomForestClassifier().fit(emb_xs, y)
#fi = pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_})
#emb_xs_filt = emb_xs.loc[fi['imp'] > .002]

# sklearn.ensemble.RandomForestRegressorI
