def RF(x_train,x_test,y_train,y_test,fold_number):
   
    from sklearn.ensemble import RandomForestClassifier
    from keras.models import load_model
    from keras.utils import to_categorical
    from numpy import dstack
    from sklearn.metrics import confusion_matrix,roc_curve,auc,classification_report, accuracy_score
    import datetime


    # load models from file
    def load_all_models(n_models):
        all_models = list()
        for i in range(n_models):
            filename = f'./results/proposed method/New/fold{fold_number}/weights/CNN_' + str(i + 1) + '.h5'
            model = load_model(filename)
            all_models.append(model)
            print('>loaded %s' % filename)
        return all_models

    
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
        model=RandomForestClassifier(n_estimators=100)
        start=datetime.datetime.now()
        model.fit(stackedX,inputy)
        end=datetime.datetime.now()
        training_time=end-start
        return model,training_time


    # make a prediction with the stacked model
    def stacked_prediction(members, model, inputX):  
        stackedX = stacked_dataset(members, inputX)
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

    # prediction
    model,training_time = fit_stacked_model(members, testX, testy)
    yhat = stacked_prediction(members, model, testX)
    predicts=yhat
    actuals=testy
    acc1 = accuracy_score(actuals, predicts)
    lst_acc.append(acc1)
    
    # calculate metrics
    fpr,tpr,thr=roc_curve(actuals,predicts)
    a=auc(fpr,tpr)
    lst_AUC.append(a)
    r=classification_report(actuals,predicts)
    lst_reports.append(r) 
    c=confusion_matrix(actuals,predicts)
    lst_matrix.append(c)
    lst_times.append(training_time)

    # print results
    results_path=f'./results/proposed method/New/fold{fold_number}/results_proposed.txt' 
    f1=open(results_path,'a')
    f1.write('\nAccuracies: '+str(lst_acc)+'\n'+'AUCs: '+str(lst_AUC)+'\n')
    f1.write('\n\nMetrics for all folds: \n\n')
    for i in range(len(lst_reports)):
        f1.write(str(lst_reports[i]))
        f1.write('\n\nTraining Time: '+str(lst_times[i]))
        f1.write('\n\nCofusion Matrix: \n'+str(lst_matrix[i])+'\n\n_____________________\n')
    f1.close()
