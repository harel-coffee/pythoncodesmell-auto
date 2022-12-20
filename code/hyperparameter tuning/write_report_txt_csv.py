#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import pandas as pd
import tuning_model_function_update
def write_report_result(model_scores,model_scores_base,model_scores_CFS,model_scores_Java_Rule,model_scores_stepwise,typesmell):

    select_method=[
        [model_scores, 'all'], 
        [model_scores_base, 'base'], 
        [model_scores_CFS, 'CFS'],
        [model_scores_Java_Rule, 'Java_Rule'],
        [model_scores_stepwise, 'stepwise'],
    ]

    for sm in select_method:
        rows = []
        for x in sm[0]:
            print(sm[1])
            report = open(typesmell + '\\' + 'tunning report\\' + x[2] + '-report_' + sm[1] + '.txt','w')
            rows_tmp = tuning_model_function_update.PrintReport(x,report)
            rows.append(rows_tmp)

        if sm[1] == 'all':
            rows_all = rows
        elif sm[1] == 'base':
            rows_base = rows
        elif sm[1] == 'CFS':
            rows_CFS = rows
        elif sm[1] == 'Java_Rule': 
            rows_Java_Rule = rows
        elif sm[1] == 'stepwise': 
            rows_stepwise = rows

        report.close()
        
    df0 = pd.DataFrame(rows_all, columns=["Model", "precision","recall","f1-score"])
    df1 = pd.DataFrame(rows_base, columns=["Model", "precision","recall","f1-score"])
    df3 = pd.DataFrame(rows_CFS, columns=["Model", "precision","recall","f1-score"])
    df7 = pd.DataFrame(rows_Java_Rule, columns=["Model", "precision","recall","f1-score"])
    df9 = pd.DataFrame(rows_stepwise, columns=["Model", "precision","recall","f1-score"])
    
    ans=pd.concat([df0,df1, df3,df7,df9], axis=1)
    ans.to_csv(typesmell + '\\' +typesmell + '-ML-tuning-update.csv',index=False)
    
    return ans


# In[ ]:




