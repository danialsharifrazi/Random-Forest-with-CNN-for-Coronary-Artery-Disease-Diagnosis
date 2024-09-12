from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from keras.models import load_model
from keras.utils import to_categorical
from numpy import dstack
import os
import glob
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import classification_report
import datetime

def RF(x_train,x_test,y_train,y_test):


    # load models from file
    def load_all_models(n_models):
        all_models = list()
        for i in range(n_models):
            filename = f'./results/CNN/New/fold{i+1}/CNN' + str(i+1) + '.h5'
            model = load_model(filename)
            all_models.append(model)
            print('>loaded %s' % filename)
        return all_models

    # create stacked model input dataset as outputs from the ensemble
    def stacked_dataset(members, inputX):
        stackX = None
        for model in members:
            yhat = model.predict(inputX, verbose=0)
            if stackX is None:
                stackX = yhat
            else:
                stackX = dstack((stackX, yhat))
        stackX = stackX.reshape((stackX.shape[0], stackX.shape[1]*stackX.shape[2]))
        return stackX


    def fit_stacked_model(members,inputX,inputy):
        stackedX=stacked_dataset(members,inputX)
        model=RandomForestClassifier(n_estimators=10)
        start=datetime.datetime.now()
        model.fit(stackedX,inputy)
        end=datetime.datetime.now()
        training_time=end-start
        return model,training_time

    # make a prediction with the stacked model
    def stacked_prediction(members, model, inputX):
        # create dataset using ensemble
        stackedX = stacked_dataset(members, inputX)
        # make a prediction
        yhat = model.predict(stackedX)
        return yhat

    lst_acc=[]
    lst_reports=[]
    lst_AUC=[]
    lst_matrix=[]
    lst_times=[]

    trainX=x_train
    testX=x_test
    trainy=y_train
    testy=y_test

    # load all models
    n_members = 10
    members = load_all_models(n_members)
    print('Loaded %d models' % len(members))
    for model in members:
        testy_enc = to_categorical(testy)
        loss, acc = model.evaluate(testX, testy_enc, verbose=0)
        
    results_path=f'./results/proposed method/New/fold{fold_number}/results_proposed.txt' 

    model,training_time = fit_stacked_model(members, testX, testy)
    yhat = stacked_prediction(members, model, testX)
    predicts=yhat
    actuals=testy
    acc1 = accuracy_score(actuals, predicts)
    lst_acc.append(acc1)

    fpr,tpr,thr=roc_curve(actuals,predicts)
    a=auc(fpr,tpr)
    lst_AUC.append(a)

    r=classification_report(actuals,predicts)
    lst_reports.append(r) 

    c=confusion_matrix(actuals,predicts)
    lst_matrix.append(c)

    lst_times.append(training_time)


    f1=open(results_path,'a')
    f1.write('\nAccuracies: '+str(lst_acc)+'\n'+'AUCs: '+str(lst_AUC)+'\n')
    f1.write('\n\nMetrics for all folds: \n\n')
    for i in range(len(lst_reports)):
        f1.write(str(lst_reports[i]))
        f1.write('\n\nTraining Time: '+str(lst_times[i]))
        f1.write('\n\nCofusion Matrix: \n'+str(lst_matrix[i])+'\n\n_________________\n')
    f1.close()

#read normals files
normals=[]
main_path='./Converted Dataset/Normal/'
main_folders=next(os.walk(main_path))[1]
for i in main_folders:
    path=main_path+i+'/'
    folders=next(os.walk(path))[1]
    for x in folders:
        new_path=path+x+'/'
        data=glob.glob(new_path+'*.jpg')
        if len(data)<1:
            indent_folders=next(os.walk(new_path))[1]
            for y in indent_folders:
                new_path=new_path+y+'/'
                data=glob.glob(new_path+'*.jpg')
        normals.extend(data)


#read sicks files
sicks=[]
main_path='./Converted Dataset/Sick/'
main_folders=next(os.walk(main_path))[1]
for i in main_folders:
    path=main_path+i+'/'
    folders=next(os.walk(path))[1]
    for x in folders:
        new_path=path+x+'/'
        data=glob.glob(new_path+'*.jpg')
        if len(data)<1:
            indent_folders=next(os.walk(new_path))[1]
            for y in indent_folders:
                new_path=new_path+y+'/'
                data=glob.glob(new_path+'*.jpg')
        sicks.extend(data)
    

#load normal files
labels_n=[]
train_data_n=[]
for id in normals:    
    img=cv2.imread(id)
    if np.max(img) !=0:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img=cv2.resize(img,(100,100))
        img=img.astype('float32')
        img=img/np.max(img)
        train_data_n.append(img)
        labels_n.append(0)

   
#load sick files
labels_s=[]
train_data_s=[]
for id in sicks:    
    img=cv2.imread(id)
    if np.max(img) !=0:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img=cv2.resize(img,(100,100))
        img=img.astype('float32')
        img=img/np.max(img)
        #img=img.flatten()
        train_data_s.append(img)
        labels_s.append(1)


train_data_n.extend(train_data_s)
labels_n.extend(labels_s)

x=np.array(train_data_n)
y=np.array(labels_n)

print('Data Loaded...!')


lst_acc=[]
lst_reports=[]
lst_AUC=[]
lst_matrix=[]
lst_times=[]

fold_number=0
from sklearn.model_selection import KFold
kfold=KFold(10,shuffle=True,random_state=0)
for train,test in kfold.split(x,y):

    x_train=x[train]
    y_train=y[train]
    x_test=x[test]
    test_labels=y[test]

    fold_number+=1
    RF(x_train,x_test,y_train,test_labels)

    