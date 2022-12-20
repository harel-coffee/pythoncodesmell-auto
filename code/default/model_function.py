#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os

from imblearn.pipeline import Pipeline as imbpipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn import metrics
from sklearn.model_selection import KFold
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn import tree


algo=[
    [KNeighborsClassifier(), 'KNeighborsClassifier'], 
    [LogisticRegression(), 'LogisticRegression'], 
    [DecisionTreeClassifier(), 'DecisionTreeClassifier'],
    [GradientBoostingClassifier(), 'GradientBoostingClassifier'],
    [RandomForestClassifier(), 'RandomForestClassifier'],
    [GaussianNB(), 'GaussianNB'],
    [MLPClassifier(), 'MLPClassifier'],
    [SVC(), 'SVM'], 
]

def print_data_model():
    for detail in algo:
        print(detail[1])
        print(detail[0].get_params())


# In[2]:


def devide_data(X,y,base,cfs,java,stepwise,smelltype):
    os.makedirs(smelltype)
    os.makedirs(smelltype+'_data\\data')
    os.makedirs(smelltype+'\\report')

    model_scores=[]

    model_scores_base=[]

    model_scores_CFS=[]

    model_scores_Java_Rule=[]
    
    model_scores_stepwise=[]
    

    for a in algo:

        os.makedirs(smelltype+ '\\' +a[1])

        report_fold_all = open(smelltype+ '\\' +a[1] + '\\' + a[1] + '-fold-all-report.txt','w')
        
        report_fold_base = open(smelltype+ '\\' +a[1] + '\\' + a[1] + '-fold-base-report.txt','w')
       
        report_fold_CFS = open(smelltype+ '\\' +a[1] + '\\' + a[1] + '-fold-CFS-report.txt','w')
     
        report_fold_Java_Rule = open(smelltype+ '\\' +a[1] + '\\' + a[1] + '-fold-Java_Rule-report.txt','w')
        
        report_fold_stepwise = open(smelltype+ '\\' +a[1] + '\\' + a[1] + '-fold-stepwise-report.txt','w')
       
        score = []

        predicted_targets = np.array([])
        actual_targets = np.array([])

        predicted_targets_base = np.array([])
        actual_targets_base = np.array([])

        predicted_targets_CFS = np.array([])
        actual_targets_CFS = np.array([])

        predicted_targets_Java_Rule = np.array([])
        actual_targets_Java_Rule = np.array([])

        predicted_targets_stepwise = np.array([])
        actual_targets_stepwise = np.array([])


        kf = KFold(n_splits=10, shuffle=True, random_state=42)  

        for fold, (train_index, test_index) in enumerate(kf.split(X), 1):

            # spilt train test
            X_train = X[train_index]
            y_train = y[train_index] 
            X_test = X[test_index]
            y_test = y[test_index]

            # start save train test to csv file
            x_train_data =  pd.DataFrame(X_train)
            y_train_data =  pd.DataFrame(y_train)
            x_test_data = pd.DataFrame(X_test)
            y_test_data = pd.DataFrame(y_test)

            train_data = pd.concat([x_train_data,y_train_data], axis=1)
            train_data.to_csv(smelltype+ '_data\\' +'data\\train_fold_' + str(fold) + '.csv',index=False)

            test_data = pd.concat([x_test_data,y_test_data], axis=1)
            test_data.to_csv(smelltype+ '_data\\' +'data\\test_fold_' + str(fold) + '.csv',index=False)
            # end


            # ***** start no feature selection *****

            y_pred = ModelPredict(a, fold, X_train, y_train, X_test, y_test, report_fold_all)

            predicted_targets = np.append(predicted_targets, y_pred)
            actual_targets = np.append(actual_targets, y_test)


            # ***** start select feature base on baseline *****
            X_train_base = X_train[:, base]
            X_test_base = X_test[:, base]

            y_pred_base = ModelPredict(a, fold, X_train_base, y_train, X_test_base, y_test, report_fold_base)

            predicted_targets_base = np.append(predicted_targets_base, y_pred_base)
            actual_targets_base = np.append(actual_targets_base, y_test)


            # ***** start select feature base on CFS *****
            X_train_CFS = X_train[:, cfs]
            X_test_CFS = X_test[:, cfs]

            y_pred_CFS = ModelPredict(a, fold, X_train_CFS, y_train, X_test_CFS, y_test, report_fold_CFS)

            predicted_targets_CFS = np.append(predicted_targets_CFS, y_pred_CFS)
            actual_targets_CFS = np.append(actual_targets_CFS, y_test)


            # ***** start select feature base on Java Rule *****
            X_train_Java_Rule = X_train[:, java]
            X_test_Java_Rule = X_test[:, java]

            y_pred_Java_Rule = ModelPredict(a, fold, X_train_Java_Rule, y_train, X_test_Java_Rule, y_test, report_fold_Java_Rule)

            predicted_targets_Java_Rule = np.append(predicted_targets_Java_Rule, y_pred_Java_Rule)
            actual_targets_Java_Rule = np.append(actual_targets_Java_Rule, y_test)


            #***** start select feature base on stepwise *****
            X_train_stepwise = X_train[:, stepwise]
            X_test_stepwise = X_test[:, stepwise]

            y_pred_stepwise = ModelPredict(a, fold, X_train_stepwise, y_train, X_test_stepwise, y_test, report_fold_stepwise)

            predicted_targets_stepwise = np.append(predicted_targets_stepwise, y_pred_stepwise)
            actual_targets_stepwise = np.append(actual_targets_stepwise, y_test)

            
        model_scores.append([actual_targets, predicted_targets, a[1]])

        model_scores_base.append([actual_targets_base, predicted_targets_base, a[1]])
        
        model_scores_CFS.append([actual_targets_CFS, predicted_targets_CFS, a[1]])
        
        model_scores_Java_Rule.append([actual_targets_Java_Rule, predicted_targets_Java_Rule, a[1]])
        
        model_scores_stepwise.append([actual_targets_stepwise, predicted_targets_stepwise, a[1]])
       

    report_fold_all.close()

    report_fold_base.close()
    
    report_fold_CFS.close()
   
    report_fold_Java_Rule.close()
    
    report_fold_stepwise.close()
       
    return model_scores,model_scores_base,model_scores_CFS,model_scores_Java_Rule,model_scores_stepwise


# In[3]:


def ModelPredict(m, fold, train_X, train_y, test_X, test_y, file):
    model = m[0]
    model.fit(train_X, train_y)
    score = model.score(test_X, test_y)
    pred_y = model.predict(test_X)
    f1score = f1_score(test_y, pred_y)
    
        
    print(f'{m[1]:20} Accuracy: {score:.04f}')        
    print(f'fold {fold}:')
    print(f'Accuracy: {model.score(test_X, test_y)}')
    print(f'f-score: {f1_score(test_y, pred_y)}')
    print(metrics.confusion_matrix(test_y, pred_y))
    print(metrics.classification_report(test_y, pred_y))
    print('-' * 100)
    
    PrintReportFold(file,m,fold,score,f1score,test_y,pred_y)
    
    return pred_y


# In[4]:


def PrintReportFold(report_fold,m,fold,score_fold,f1score_fold,y_test_fold,y_pred_fold):
    report_fold.write(f'{m[1]:20} \n')
    report_fold.write(f'fold {fold}: \n')
    report_fold.write(f'Accuracy: {score_fold} \n')
    report_fold.write(f'f-score: {f1score_fold} \n')
    report_fold.write(str(metrics.confusion_matrix(y_test_fold, y_pred_fold)) + '\n')
    report_fold.write(metrics.classification_report(y_test_fold, y_pred_fold))
    report_fold.write('-' * 100 + '\n')


# In[5]:


def PrintReport(i,report):
    
    rows = []

    print(f'{i[2]:20}')
    print(metrics.confusion_matrix(i[0], i[1])) 
    print(metrics.classification_report(i[0], i[1]))
    
    report.write(f'{i[2]:20}\n')
    report.write(str(metrics.confusion_matrix(i[0], i[1])) + '\n') 
    report.write(metrics.classification_report(i[0], i[1]))
    
    report_to_csv = pd.DataFrame(metrics.classification_report(i[0], i[1], output_dict=True)).transpose()
    rows =[i[2],report_to_csv['precision'][1],report_to_csv['recall'][1],report_to_csv['f1-score'][1]]
    
    return rows

