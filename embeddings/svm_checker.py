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


#start = 5634
#end = 5824

# ignored ??? here i suppose
#y, x = svm_read_problem('./embeddings_bgnoised_tot/embeddings_libsvm.data')
y, x = svm_read_problem('./embeddings_bgnoised_tot/forward_libsvm.data')
start = 5628-1
end = 5818-1


include_val = False

#y, x = svm_read_problem('./embeddings_bgnoised/embeddings_libsvm.data')
#start = 1-1
#end = 191-1



totcount = 0
ccount = 0

#print(int(y[5823]))
print(type(y))
ccount_label = dict()
totcount_label = dict()
for li in range(start, end):
    if li == end-1:
        if include_val:
            m = svm_train(y[:li], x[:li], '-c 4 -q')
        else:
            m = svm_train(y[start:li], x[start:li], '-c 4 -q')
    else:
        if include_val:
            m = svm_train(y[:li]+y[(li+1):], x[:li]+x[(li+1):], '-c 4 -q')
        else:
            m = svm_train(y[start:li]+y[(li+1):], x[start:li]+x[(li+1):], '-c 4 -q')

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

    print(li, "[",int(y[li]),"]", ": plabel,pacc: ",p_label, " ", p_acc)


print(f"Final prediction accuracy: {ccount}/{totcount}={ccount/totcount}")
for k in ccount_label.keys():
    print(f"{k}: {ccount_label[k]}/{totcount_label[k]}={ccount_label[k]/totcount_label[k]}")

print(f"Final prediction accuracy: {ccount}/{totcount}={ccount/totcount}")
