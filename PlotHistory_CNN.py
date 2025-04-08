
def NetPlot(net_history,fold_number):
    import matplotlib.pyplot as plt
    history=net_history.history
    losses=history['loss']
    val_losses=history['val_loss']
    accuracies=history['acc']
    val_accuracies=history['val_acc']

    path1=f'./results/CNN/fold{fold_number}/Losses.txt'
    path2=f'./results/CNN/fold{fold_number}/ValLosses.txt'
    path3=f'./results/CNN/fold{fold_number}/Acc.txt'
    path4=f'./results/CNN/fold{fold_number}/ValAcc.txt'

    f1=open(path1,'a')
    f2=open(path2,'a')
    f3=open(path3,'a')
    f4=open(path4,'a')

    f1.write(str(losses)+'\n')
    f2.write(str(val_losses)+'\n')
    f3.write(str(accuracies)+'\n')
    f4.write(str(val_accuracies)+'\n')

    plt.figure(f'Loss Diagram_{fold_number}',dpi=200)
    plt.title('Loss of Deep Neural Network')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(losses,color='black')
    plt.plot(val_losses,color='red')
    plt.legend(['Train Data','Validation Data'])
    plt.savefig(f'./results/CNN/fold{fold_number}/Loss Diagram.png')

    plt.figure(f'Accuracy Diagram_{fold_number}',dpi=200)
    plt.title('Accuracy of Deep Neural Network')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(accuracies,color='black')
    plt.plot(val_accuracies,color='red')
    plt.legend(['Train Data','Validation Data'])       
    plt.savefig(f'./results/CNN/fold{fold_number}/Accuracy Diagram.png')
    


