#https://stats.stackexchange.com/questions/61328/libsvm-data-format
from libsvm.svmutil import *
#y, x = [1,-1], [[1,0,1], [-1,0,-1]]
#prob = svm_problem(y, x)
#param = svm_parameter('-t 0 -c 4 -b 1')
#m = svm_train(prob, param)
#p_label, p_acc, p_val = svm_predict(y, x, m, '-b 1')
#print("plabel,pacc,pval: ",p_label, " ", p_acc, " ", p_val)
#ACC, MSE, SCC = evaluations(y, p_label)
#print("acc,mse,scc: ",ACC, " ",MSE, " ",SCC)

y, x = svm_read_problem('heart_scale')
m = svm_train(y[:200], x[:200], '-c 4')
p_label, p_acc, p_val = svm_predict(y[200:], x[200:], m)
print("plabel,pacc,pval: ",p_label, " ", p_acc, " ", p_val)


