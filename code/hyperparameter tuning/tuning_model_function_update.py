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

from sklearn.model_selection import RandomizedSearchCV

from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingGridSearchCV

grid_knn = { 
    'n_neighbors' : list(range(1,10)),
    'leaf_size' : list(range(1,50)),
    'p':[1,2]
}

grid_dt = { 
    'max_depth' : [None,2,4,6,8,10,12],
    'criterion' : ['gini', 'entropy'],
}

grid_mlp = { 
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}

grid_gbt = { 
   'learning_rate': [0.1,0.01],
    'n_estimators' : [100,500,1000],
    'max_depth'    : [3,4,6,8,10,12]
}

grid_svm = { 
    'C':list(range(-1,11)),

}

grid_nb = { 
    'var_smoothing': [1e-11, 1e-10, 1e-9]
}

grid_lr = { 
      'C':list(range(-1,11)), 
      'penalty' : ['l1', 'l2'],
      'max_iter': [20, 50, 100, 200, 500, 1000],                      
      'solver': ['lbfgs', 'liblinear'],   
      'class_weight': [None,'balanced']
}

grid_rf = { 
    'n_estimators': [5, 20, 50, 100],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
}




# In[2]:


algo=[
    [KNeighborsClassifier(), 'KNeighborsClassifier tuning',grid_knn], 
    [LogisticRegression(), 'LogisticRegression tuning',grid_lr], 
    [DecisionTreeClassifier(), 'DecisionTreeClassifier tuning',grid_dt],
    [GradientBoostingClassifier(), 'GradientBoostingClassifier tuning',grid_gbt],
    [RandomForestClassifier(), 'RandomForestClassifier',grid_rf],
    [GaussianNB(), 'GaussianNB tuning',grid_nb],
    [MLPClassifier(), 'MLPClassifier tuning',grid_mlp],
    [SVC(), 'SVM tuning',grid_svm], 

]

def print_data_model():
    for detail in algo:
        print(detail[1])
        print(detail[0].get_params())
        
print_data_model()


# In[3]:


def tunning_model(path,base,cfs,java,stepwise,smelltype):
    
    os.makedirs(smelltype)
    os.makedirs(smelltype + '\\' +'tunning')

    model_scores=[]
    model_scores_base=[]
    model_scores_CFS=[]
    model_scores_Java_Rule=[]
    model_scores_stepwise=[]

    for a in algo:

        os.makedirs(smelltype + '\\' +'tunning\\'+a[1])

        report_fold_all = open(smelltype + '\\' +'tunning\\' + a[1] + '\\' + a[1] + '-fold-all-report.txt','w')
        report_fold_base = open(smelltype + '\\' +'tunning\\' + a[1] + '\\' + a[1] + '-fold-base-report.txt','w')
        report_fold_CFS = open(smelltype + '\\' +'tunning\\' + a[1] + '\\' + a[1] + '-fold-CFS-report.txt','w')
        report_fold_Java_Rule = open(smelltype + '\\' +'tunning\\' + a[1] + '\\' + a[1] + '-fold-Java_Rule-report.txt','w')
        report_fold_stepwise = open(smelltype + '\\' +'tunning\\' + a[1] + '\\' + a[1] + '-fold-stepwise-report.txt','w')
       

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

        
        for fold in range(1,11):
            
            train = pd.read_csv(path + 'train_fold_' + str(fold) + ".csv")
            test = pd.read_csv(path + 'test_fold_' + str(fold) + ".csv")
  
            
            X_train = train.iloc[:,:-1].to_numpy()
            X_test = test.iloc[:,:-1].to_numpy()
            y_train = train.iloc[:,-1].to_numpy()
            y_test = test.iloc[:,-1].to_numpy()
            
            
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


# In[4]:


def ModelPredict(m, fold, train_X, train_y, test_X, test_y, file):
    
    search = GridSearchCV(estimator=m[0], param_grid=m[2], cv= 10, n_jobs=-1,scoring='accuracy')
    
    model = search.fit(train_X, train_y)
    
    best_model = model.best_estimator_
    score = best_model.score(test_X, test_y)
    pred_y = best_model.predict(test_X)
    f1score = f1_score(test_y, pred_y)
    
        
    print(f'{m[1]:20} Accuracy: {score:.04f}')        
    print(f'fold {fold}:')
    print(best_model)
    print(f'Accuracy: {best_model.score(test_X, test_y)}')
    print(f'f-score: {f1_score(test_y, pred_y)}')
    print(metrics.confusion_matrix(test_y, pred_y))
    print(metrics.classification_report(test_y, pred_y))
    print('-' * 100)
    
    PrintReportFold(file,m,fold,score,f1score,test_y,pred_y,model.best_params_)
    
    return pred_y


# In[5]:



def PrintReportFold(report_fold,m,fold,score_fold,f1score_fold,y_test_fold,y_pred_fold,best_params):
    report_fold.write(f'{m[1]:20} \n')
    report_fold.write(f'fold {fold}: \n')
    report_fold.write(f'best_params_ {best_params}: \n')
    report_fold.write(f'Accuracy: {score_fold} \n')
    report_fold.write(f'f-score: {f1score_fold} \n')
    report_fold.write(str(metrics.confusion_matrix(y_test_fold, y_pred_fold)) + '\n')
    report_fold.write(metrics.classification_report(y_test_fold, y_pred_fold))
    report_fold.write('-' * 100 + '\n')


# In[6]:


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


# In[ ]:




