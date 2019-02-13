
# coding: utf-8

# In[ ]:


# coding: utf-8

# In[ ]:

#Importing libraries
import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import itertools
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from time import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.model_selection import train_test_split,cross_val_score,learning_curve,validation_curve 
from sklearn.metrics import fbeta_score , accuracy_score,roc_auc_score,make_scorer,roc_auc_score,roc_curve,scorer,f1_score
import statsmodels.api as sm
from sklearn.metrics import precision_score,recall_score
from yellowbrick.classifier import DiscriminationThreshold
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,label=True):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    if label == True:    
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
    plt.tight_layout()


def telecom_churn_prediction(algorithm,name,X_train, X_test, y_train, y_test,cols,cf=None,plot=False,threshold=False) : 
    #model
    start = time() # Get start time
    algorithm.fit(X_train,y_train)
    end = time() # Get end time
    # Calculate the training time
    train_time = round(end-start,4)
    
    #predict
    start = time() # Get start time
    predictions_test   = algorithm.predict(X_test)
    end = time() # Get end time
    # Calculate the training time
    pred_time = round(end-start,4)
    
    predictions_train   = algorithm.predict(X_train)
    probabilities = algorithm.predict_proba(X_test)
    
    #coeffs
    if cf != None:
        if cf == "coefficients" :
            coefficients  = pd.DataFrame(algorithm.coef_.ravel())
        elif cf == "features" :
            coefficients  = pd.DataFrame(algorithm.feature_importances_)

        column_df     = pd.DataFrame(cols)
        coef_sumry    = (pd.merge(coefficients,column_df,left_index= True,
                                  right_index= True, how = "left"))
        coef_sumry.columns = ["coefficients","features"]
        coef_sumry    = coef_sumry.sort_values(by = "coefficients",ascending = False)
    
    print (algorithm)
    print ("\n Classification report : \n",classification_report(y_test,predictions_test))   
    #confusion matrix
    conf_matrix = confusion_matrix(y_test,predictions_test)
    
    #roc_auc_score
    model_roc_auc = roc_auc_score(y_test,predictions_test) 
    print ('train')
    print ("Accuracy   Score : ",accuracy_score(y_train,predictions_train))
    print ("Area under curve : ",roc_auc_score(y_train,predictions_train),"\n")
    print ('test')
    print ("Accuracy   Score :",accuracy_score(y_test,predictions_test))    
    print ("Area under curve : ",model_roc_auc,"\n")
    fpr,tpr,thresholds = roc_curve(y_test,probabilities[:,1])

    accuracy     = accuracy_score(y_test,predictions_test)
    recallscore  = recall_score(y_test,predictions_test)
    precision    = precision_score(y_test,predictions_test)
    roc_auc_train      = roc_auc_score(y_train,predictions_train)
    roc_auc_test      = roc_auc_score(y_test,predictions_test)
    f1score      = f1_score(y_test,predictions_test) 
    result = pd.DataFrame({"Model"           : [name],
                       "Accuracy_score"  : [accuracy],
                       "Recall_score"    : [recallscore],
                       "Precision"       : [precision],
                       "f1_score"        : [f1score],
                       "Area_under_curve(train)": [roc_auc_train],    
                       "Area_under_curve(test)": [roc_auc_test],
                       "train_time"    : [train_time],
                        'pred_time'     :[pred_time]
                  })
    if cf != None: 
        plt.figure(figsize = (12,8))    
        #plot confusion matrix    
        plt.subplot(221)
        plt.grid(b=None) #無網格
        plot_confusion_matrix(conf_matrix,["Not churn","Churn"])
        plt.subplot(222)
        #plot roc curve
        plt.plot(fpr, tpr, label="ROC Curve" )
        plt.title('Receiver operating characteristic')
        plt.xlabel("false positive rate")
        plt.ylabel("true positive rate (recall)")  
        #plot coeffs
        sns.set(font_scale=1)
        plt.subplot(212)
        plt.title('Feature Importances')
        plt.xticks(rotation='90')
        sns.barplot(coef_sumry['features'],coef_sumry['coefficients'])
        plt.subplots_adjust(top=1.2, bottom=0.2, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)
        if threshold == True:
        #plot threshold
            plt.figure(figsize = (14,4)) 
            visualizer = DiscriminationThreshold(algorithm)
            visualizer.fit(X_train,y_train)
            visualizer.poof()  
    elif cf == None:
        plt.figure(figsize = (12,4))    
        #plot confusion matrix    
        plt.subplot(121)
        plt.grid(b=None) #無網格
        plot_confusion_matrix(conf_matrix,["Not churn","Churn"])
        plt.subplot(122)
        #plot roc curve
        plt.plot(fpr, tpr, label="ROC Curve" )
        plt.title('Receiver operating characteristic')
        plt.xlabel("false positive rate")
        plt.ylabel("true positive rate (recall)")  
        plt.subplots_adjust(top=1.2, bottom=0.2, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)      
    return result



scorer = make_scorer(fbeta_score, beta=0.5)
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("fbeta_score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs,scoring=scorer, train_sizes=train_sizes,random_state =0)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Validation score")

    plt.legend(loc="best")
    print("{} test mean scores = {} ".format(estimator.__class__.__name__,test_scores_mean[4]))
    return plt


