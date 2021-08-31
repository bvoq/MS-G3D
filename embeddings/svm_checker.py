
#https://stats.stackexchange.com/questions/61328/libsvm-data-format
from libsvm.svmutil import *
import optuna

# How to use
# module load python/3.7.1
# optuna create-study --study-name "study1" --storage "sqlite:///study1.db"
# bsub -n 1 -R "rusage[mem=512]" -W 23:00 python3 svm_checker.py


#y, x = [1,-1], [[1,0,1], [-1,0,-1]]
#prob = svm_problem(y, x)
#param = svm_parameter('-t 0 -c 4 -b 1')
#m = svm_train(prob, param)
#p_label, p_acc, p_val = svm_predict(y, x, m, '-b 1')
#print("plabel,pacc,pval: ",p_label, " ", p_acc, " ", p_val)
#ACC, MSE, SCC = evaluations(y, p_label)
#print("acc,mse,scc: ",ACC, " ",MSE, " ",SCC)


#start = 5634
#end = 5824

# ignored ??? here i suppose
dataset = "embeddings_bgasclass_tot"
dataset = "embeddings_tottrained"
dataset = "embeddings_bgnoised_tot"
#y, x = svm_read_problem('./'+dataset+'/embeddings_libsvm.data')
y, x = svm_read_problem('./'+dataset+'/forward_libsvm.data')
start = 5628-1
end = 5818-1


include_val = False

#y, x = svm_read_problem('./embeddings_bgnoised/embeddings_libsvm.data')
#start = 1-1
#end = 191-1

def objective(trial):
    #classifier = trial.suggest_categorical('classifier', ['0','3','4'])
    #-s svm_type : set type of SVM (default 0)
    #0 -- C-SVC
    #1 -- nu-SVC
    #2 -- one-class SVM
    #3 -- epsilon-SVR
    #4 -- nu-SVR
    #kernel = trial.suggest_categorical('kernel', ['0','1','2','3'])
    classifier = '0'
    kernel = '0'
    #-t kernel_type : set type of kernel function (default 2)
    #0 -- linear: u'*v
    #1 -- polynomial: (gamma*u'*v + coef0)^degree
    #2 -- radial basis function: exp(-gamma*|u-v|^2)
    #3 -- sigmoid: tanh(gamma*u'*v + coef0)

    cost = trial.suggest_loguniform('cost', 1e-1,1e2)
    #nu = trial.suggest_loguniform('nu',1e-5,1e3)
    #coef0 = trial.suggest_loguniform('coef0',1e-9,1e3)


    #cost = trial.suggest_uniform('cost',0,5)
    #nu = trial.suggest_loguniform('nu',1e-5,1)
    #eps = trial.suggest_loguniform('eps',1e-7,1e-1)

    #-g gamma : set gamma in kernel function (default 1/num_features)
    #-r coef0 : set coef0 in kernel function (default 0)
    #-c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
    #-n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
    #-p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)
    #-m cachesize : set cache memory size in MB (default 100)
    #-e epsilon : set tolerance of termination criterion (default 0.001)

    print("class ", classifier, " ker ", kernel, " cost ", cost )

    totcount = 0
    ccount = 0

    print(type(y))
    ccount_label = dict()
    totcount_label = dict()
    for li in range(start, end):
        if li == end-1:
            if include_val:
                m = svm_train(y[:li], x[:li], '-s '+classifier+' -t '+kernel+' -q')
            else:
                m = svm_train(y[start:li], x[start:li], '-s '+classifier+' -t '+kernel+' -q')
        else:
            if include_val:
                m = svm_train(y[:li]+y[(li+1):], x[:li]+x[(li+1):], '-s '+classifier+' -t '+kernel+' -q')
            else:
                m = svm_train(y[start:li]+y[(li+1):], x[start:li]+x[(li+1):], '-s '+classifier+' -t '+kernel+' -q')
    
        p_label, (p_acc, lmsq_e, blub), p_val = svm_predict([y[li]], [x[li]], m)
        la = int(y[li])
        #if (p_acc >= 50): # either 100 or 0 for classification
        if int(p_label[0]) == la:
            if not la in ccount_label:
                ccount_label[la] = 0
            ccount_label[la] += 1
            ccount += 1
    
        totcount += 1
        if not la in totcount_label:
            totcount_label[la] = 0
        totcount_label[la] += 1


    return ccount/totcount




#print(int(y[5823]))
#print(type(y))
#ccount_label = dict()
#totcount_label = dict()
#for li in range(start, end):
#    if li == end-1:
#        if include_val:
#            m = svm_train(y[:li], x[:li], '-c 4 -q')
#        else:
#            m = svm_train(y[start:li], x[start:li], '-c 4 -q')
#    else:
#        if include_val:
#            m = svm_train(y[:li]+y[(li+1):], x[:li]+x[(li+1):], '-c 4 -q')
#        else:
#            m = svm_train(y[start:li]+y[(li+1):], x[start:li]+x[(li+1):], '-c 4 -q')
#
#    p_label, (p_acc, lmsq_e, blub), p_val = svm_predict([y[li]], [x[li]], m)
#    la = int(y[li])
#    #if (p_acc >= 50): # either 100 or 0 for classification
#    if int(p_label[0]) == la:
#        if not la in ccount_label:
#            ccount_label[la] = 0
#        ccount_label[la] += 1
#        ccount += 1
#
#    totcount += 1
#    if not la in totcount_label:
#        totcount_label[la] = 0
#    totcount_label[la] += 1
#
#    print(li, "[",int(y[li]),"]", ": plabel,pacc: ",p_label, " ", p_acc)




#print(f"Final prediction accuracy: {ccount}/{totcount}={ccount/totcount}")
#for k in ccount_label.keys():
#    print(f"{k}: {ccount_label[k]}/{totcount_label[k]}={ccount_label[k]/totcount_label[k]}")
#
#print(f"Final prediction accuracy: {ccount}/{totcount}={ccount/totcount}")

#study = optuna.create_study()
#study.optimize(objective, n_trials=100)
#print('Best value: {} (params: {})\n'.format(study.best_value, study.best_params))


study = optuna.create_study(direction="maximize") #optuna.load_study(study_name='study1', storage='sqlite:///study1.db')
study.optimize(objective, n_trials=4)
print('Best value: {} (params: {})\n'.format(study.best_value, study.best_params))

