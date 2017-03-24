import cPickle as pick

def unpickle(path):
    f = open(path,'rb')
    dict1 = pick.load(f)
    f.close()
    return dict1


data_Train_cat_0 = []
labels_Train_cat_0 = []
data_Train_cat_1 = []
labels_Train_cat_1 = []
path = "../cifar10/"
for i in range(1,6):
    now_Path = path+'data_batch_'+str(i)+'.bin'
    dict1 = unpickle(now_Path)
    it  = 0
    while it < len(dict1['data']):
        data_Train_cat_0.append(dict1['data'][it])
        labels_Train_cat_0.append(dict1['labels'][it])
        it += 1
    f = file(now_Path + "alt", "wb")
    pick.dump(data_Train_cat_0, f)
    pick.dump(labels_Train_cat_0, f)
    f.close()
