import glob
import os
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import auc
from sklearn.metrics import classification_report
import datetime

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
        train_data_s.append(img)
        labels_s.append(1)

train_data_n.extend(train_data_s)
labels_n.extend(labels_s)

x_data=np.array(train_data_n)
y_data=np.array(labels_n)

lst_loss=[]
lst_acc=[]
lst_reports=[]
lst_AUC=[]
lst_matrix=[]
lst_times=[]


fold_number=1
kfold=KFold(5,shuffle=True,random_state=0)
for train,test in kfold.split(x_data,y_data):

    x_train=x_data[train]
    train_labels=y_data[train]
    x_test=x_data[test]
    test_labels=y_data[test]

    from keras.utils import np_utils
    y_train=np_utils.to_categorical(train_labels)
    y_test=np_utils.to_categorical(test_labels)

    x_train,x_valid,y_train,y_valid=train_test_split(x_train,y_train,test_size=0.3,random_state=0)

    from keras.models import Sequential
    from keras.layers import Dense,Flatten,Conv1D,Dropout
    from keras.optimizers import SGD
    from keras.losses import binary_crossentropy


    model=Sequential()
    model.add(Conv1D(32,3,padding='same',activation='relu',strides=2,input_shape=(100,100)))
    model.add(Dropout(0.2))
    model.add(Conv1D(64,3,padding='same',activation='relu',strides=2))
    model.add(Dropout(0.2))
    model.add(Conv1D(128,3,padding='same',activation='relu',strides=2))
    model.add(Dropout(0.2))
    model.add(Conv1D(256,3,padding='same',activation='relu',strides=1))
    model.add(Dropout(0.2))
    model.add(Conv1D(256,3,padding='same',activation='relu',strides=1))
    model.add(Dropout(0.2))
    model.add(Conv1D(256,3,padding='same',activation='relu',strides=1))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(256,activation='relu'))
    model.add(Dense(128,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(2,activation='sigmoid'))

    model.compile(optimizer=SGD(),loss=binary_crossentropy,metrics=['accuracy'])
        
    start=datetime.datetime.now()
    net_history=model.fit(x_train, y_train, batch_size=256, epochs=30 ,validation_data=[x_valid,y_valid])
    end=datetime.datetime.now()

    model.save(f'./results/CNN/fold{fold_number}/CNN{fold_number}.h5')


    test_loss, test_acc=model.evaluate(x_test,y_test)
    lst_acc.append(test_acc)
    lst_loss.append(test_loss)



    predicts=model.predict(x_test)
    predicts=predicts.argmax(axis=1)
    actuals=y_test.argmax(axis=1)

    fpr,tpr,thr=roc_curve(actuals,predicts)
    a=auc(fpr,tpr)
    lst_AUC.append(a)

    r=classification_report(actuals,predicts)
    lst_reports.append(r)

    c=confusion_matrix(actuals,predicts)
    lst_matrix.append(c)

    training_time=end-start
    lst_times.append(training_time)


    import PlotHistory_CNN
    PlotHistory_CNN.NetPlot(net_history,fold_number)
    fold_number+=1



path=f'./results/CNN/CNN_Results.txt' 
f1=open(path,'a')
f1.write('\nAccuracies: '+str(lst_acc)+'\nLosses: '+str(lst_loss)+'\nAUCs: '+str(lst_AUC)+'\n')
f1.write('\n\nMetrics for all Folds: \n\n')
for i in range(len(lst_reports)):
    f1.write(str(lst_reports[i]))
    f1.write('\n\nTraining Time: '+str(lst_times[i]))
    f1.write('\n\nCofusion Matrix: \n'+str(lst_matrix[i])+'\n\n________________________\n')
f1.close()





