import numpy as np
import scipy as sp
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Plotting options
mpl.style.use('ggplot')
sns.set(style='whitegrid')

loan_good=pd.DataFrame()
for chunk in pd.read_csv(r'D:\Adv SAS\lending-club\accepted_2007_to_2018Q3.csv', chunksize=50000, low_memory= False):
    loan_good = pd.concat([loan_good, chunk], ignore_index=True)

loans = loan_good.loc[loan_good['loan_status'].isin(['Fully Paid', 'Charged Off'])]
loans=loans.set_index('id')
#Drop columns which has 75% or more missing values
missing_fractions = loans.isnull().mean().sort_values(ascending=False)
#plt.figure(figsize=(6,3), dpi=90)
#missing_fractions.plot.hist(bins=20)
#plt.title('Histogram of Feature Incompleteness')
#plt.xlabel('Fraction of data missing')
#plt.ylabel('Feature count')
drop_list = sorted(list(missing_fractions[missing_fractions > 0.25].index))
len(drop_list)
loans.drop(labels=drop_list, axis=1, inplace=True)

#Fix date
loans['issue_d'].isnull().any()
import datetime
loans['issue_d']=pd.to_datetime(loans['issue_d'])
loans['earliest_cr_line'] = pd.to_datetime(loans['earliest_cr_line'])
#Keep Variables that will be known when approving loan
keep_list = ['addr_state', 'annual_inc', 'application_type', 'dti', 'earliest_cr_line', 'emp_length', 'emp_title', 'fico_range_high', 'fico_range_low', 'home_ownership', 'id', 'initial_list_status', 'installment', 'int_rate', 'issue_d', 'loan_amnt', 'loan_status', 'mort_acc', 'open_acc', 'pub_rec', 'pub_rec_bankruptcies', 'purpose', 'revol_bal', 'revol_util', 'sub_grade', 'term', 'total_acc', 'verification_status']
drop_list = [col for col in loans.columns if col not in keep_list]
loans.drop(labels=drop_list, axis=1, inplace=True)
#chunk.drop(labels=drop_list, axis=1, inplace=True)
#Feature Engineering
q= loans.groupby('emp_title').size()
q=q/len(loans)
loans["emp_ttl"]= loans.emp_title.map(q)
loans=loans.drop('emp_title', axis=1)

def status_to_numeric(y):
    if y=='Fully Paid':
        return 0
    if y=='Charged Off':
        return 1
loans['loan_status'] = loans['loan_status'].apply(status_to_numeric)    #target column

def grade_to_numeric(yy):
    if yy=='A1':
        return 1
    if yy=='A2':
        return 2
    if yy=='A3':
        return 3
    if yy=='A4':
        return 4
    if yy=='A5':
        return 5
    if yy=='B1':
        return 6
    if yy=='B2':
        return 7
    if yy=='B3':
        return 8
    if yy=='B4':
        return 9
    if yy=='B5':
        return 10
    if yy=='C1':
        return 11
    if yy=='C2':
        return 12
    if yy=='C3':
        return 13
    if yy=='C4':
        return 14
    if yy=='C5':
        return 15
    if yy=='D1':
        return 16
    if yy=='D2':
        return 17
    if yy=='D3':
        return 18
    if yy=='D4':
        return 19
    if yy=='D5':
        return 20
    if yy=='E1':
        return 21
    if yy=='E2':
        return 22
    if yy=='E3':
        return 23
    if yy=='E4':
        return 24
    if yy=='E5':
        return 25
    if yy=='F1':
        return 26
    if yy=='F2':
        return 27
    if yy=='F3':
        return 28
    if yy=='F4':
        return 29
    if yy=='F5':
        return 30
    if yy=='G1':
        return 31
    if yy=='G2':
        return 32
    if yy=='G3':
        return 33
    if yy=='G4':
        return 34
    if yy=='G5':
        return 35
loans['Grade'] = loans['sub_grade'].apply(grade_to_numeric)   
loans = loans.drop('sub_grade', axis=1)

def length_to_numeric(yl):
    if yl=='1 year':
        return 1
    if yl=='2 years':
        return 2
    if yl=='3 years':
        return 3
    if yl=='4 years':
        return 4
    if yl=='5 years':
        return 5
    if yl=='6 years':
        return 6
    if yl=='7 years':
        return 7
    if yl=='8 years':
        return 8
    if yl=='9 years':
        return 9
    if yl=='10 years':
        return 10
    if yl=='< 1 year':
        return 0.5
    if yl=='10+ years':
        return 11
loans['em_length'] = loans['emp_length'].apply(length_to_numeric)
loans = loans.drop('emp_length', axis=1)

loans['credit_length']=loans['issue_d']-loans['earliest_cr_line']
loans['credit_length']=loans['credit_length'].dt.days

loans['term2']=loans['term'].str.extract("(\d*\.?\d+)", expand=True).astype(int)
loans=loans.drop('term',axis=1)
loans['emp_len']=loans['em_length'].apply(lambda x: 0.5 if x==-0.5 else x)
loans=loans.drop('em_length',axis=1)

loans['earliest_cr_yr']=loans.earliest_cr_line.dt.year
loans=loans[loans.earliest_cr_yr < 2018]
loans=loans.drop(['issue_d','earliest_cr_line', 'earliest_cr_yr'],axis=1)

loans= pd.read_csv(r'D:\Adv SAS\lending-club\accepted\Loans_processed.csv')
loans=loans.drop('Unnamed: 0', axis=1).set_index('id')
loans= loans.dropna()
#Standardizing
from sklearn.preprocessing import standardscaler
SC= standardscaler()
num_cols = loans.columns[loans.dtypes.apply(lambda c: np.issubdtype(c, np.number))]
num_cols=num_cols.drop(['loan_status'])
loans[num_cols] = SC(loans[num_cols])

#Resizing the dataframe
def reduce_mem_usage(props):
    start_mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in. 
    for col in props.columns:
        if props[col].dtype != object:  # Exclude strings
            
            # Print current column type
            print("******************************")
            print("Column: ",col)
            print("dtype before: ",props[col].dtype)
            
            # make variables for Int, max and min
            IsInt = False
            mx = props[col].max()
            mn = props[col].min()
            
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(props[col]).all(): 
                NAlist.append(col)
                props[col].fillna(mn-1,inplace=True)  
                   
            # test if column can be converted to an integer
            asint = props[col].fillna(0).astype(np.int64)
            result = (props[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            
            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)    
            
            # Make float datatypes 32 bit
            else:
                props[col] = props[col].astype(np.float32)
            
            # Print new column type
            print("dtype after: ",props[col].dtype)
            print("******************************")
    
    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return props, NAlist

props, NAlist = reduce_mem_usage(loans)
print("_________________")
print("")
print("Warning: the following columns have missing values filled with 'df['column_name'].min() -1': ")
print("_________________")
print("")
print(NAlist)
#props.to_csv(r'D:\Adv SAS\lending-club\accepted\Loans_processed.csv',index=True)
loans= props.dropna()
#ln=pd.get_dummies(loans, columns=[ 'application_type','home_ownership', 'verification_status','addr_state', 'initial_list_status', 'purpose', ], drop_first=True)

#xx= xx.drop('loan_status', axis=1)
#y= loans['loan_status']
#XX['issue_yr']=XX.issue_d.dt.year
#XX['issue_mth']=XX.issue_d.dt.month

#ln=ln.dropna()

#splitting 3-way & 2-way
train, validate, test = np.split(loans.sample(frac=1), [int(.6*len(loans)), int(.8*len(loans))])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(loans.drop('loan_status', axis=1),loans.loan_status, test_size=0.30, random_state=420)
#Smoteing
from imblearn.over_sampling import SMOTENC
smote_nc = SMOTENC(categorical_features=[3,5,6,7,16,17,23,24], random_state=420)
X, y = smote_nc.fit_resample(X_train, y_train)
Xx, Yy = smote_nc.fit_resample(train.drop('loan_status', axis=1),train.loan_status)
np.save(r'D:\Adv SAS\lending-club\accepted\Loans_balanced.npy', X)
np.save(r'D:\Adv SAS\lending-club\accepted\Loans_status.npy', y)
np.save(r'D:\Adv SAS\lending-club\accepted\Loans_balanced2.npy', Xx)
np.save(r'D:\Adv SAS\lending-club\accepted\Loans_status2.npy', Yy)

#labeling categorical for using in lightgbm
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

xtr= pd.DataFrame(X)
xtr.columns= X_train.columns
xtr2= pd.DataFrame(Xx)
xtr2.columns= X_train.columns

class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)

X1= MultiColumnLabelEncoder(columns = [ 'application_type','home_ownership', 'verification_status','addr_state', 'initial_list_status', 'purpose']).fit_transform(xtr).values
Xt1= MultiColumnLabelEncoder(columns = [ 'application_type','home_ownership', 'verification_status','addr_state', 'initial_list_status', 'purpose']).fit_transform(X_test).values
X2= MultiColumnLabelEncoder(columns = [ 'application_type','home_ownership', 'verification_status','addr_state', 'initial_list_status', 'purpose']).fit_transform(xtr2).values
Xv= MultiColumnLabelEncoder(columns = [ 'application_type','home_ownership', 'verification_status','addr_state', 'initial_list_status', 'purpose']).fit_transform(validate.drop('loan_status',axis=1)).values
Xt2= MultiColumnLabelEncoder(columns = [ 'application_type','home_ownership', 'verification_status','addr_state', 'initial_list_status', 'purpose']).fit_transform(test.drop('loan_status',axis=1)).values
#xtr.to_csv(r'C:\Users\Alvi Mahmud\Desktop\Adv SAS\lending-club\accepted\Loans_balanced.csv',index=False)

#Lightgbm with early stopping
import lightgbm as lgb
d_train = lgb.Dataset(X2, label=Yy, categorical_feature=[3,5,6,7,16,17,23,24], free_raw_data=False)
d_val = lgb.Dataset(Xv, label=validate.loan_status.values, reference=d_train)
param = {'num_leaves':1000, 'objective':'binary','max_depth':30,'learning_rate':.1,'max_bin':800,
         'min_data_in_leaf': 500, 'task':'predict', 'feature_fraction':0.7,"bagging_fraction" : 0.6,
        "bagging_freq" : 1, "bagging_seed" : 2018,
        "verbosity" : 4, "min_child_samples":30}
param['metric'] = ['auc', 'binary_logloss']
clf = lgb.train(param, d_train, 2000,valid_sets=d_val, early_stopping_rounds=50)
y_pred=clf.predict(Xt2)
#convert into binary values
for i in range(len(Xt2)):
    if y_pred[i]>=.5:       # setting threshold to .5
       y_pred[i]=1
    else:  
       y_pred[i]=0
from sklearn.metrics import confusion_matrix
cmbl = confusion_matrix(test.loan_status, y_pred)
#Accuracy
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_pred,test.loan_status)

#Lightgbm with cross-validation and early stopping
d_train2 = lgb.Dataset(X1, label=y, categorical_feature=[3,5,6,7,16,17,21,23], free_raw_data=False)
param2 = {'num_leaves':1200, 'objective':'binary','max_depth':10,'learning_rate':.0005,'max_bin':800,
         'min_data_in_leaf': 500, 'task':'predict', 'boosting':'dart', 'feature_fraction':0.8,"bagging_fraction" : 0.7,
        "bagging_freq" : 1, "bagging_seed" : 2018,
        "verbosity" : -1, "min_child_samples":20}
param2['metric'] = ['auc', 'binary_logloss']
clf2 = lgb.cv(param, d_train, 1000,nfold=10,early_stopping_rounds=40)
y_pred2=clf2.predict(Xt1)
#convert into binary values
for i in range(len(Xt1)):
    if y_pred2[i]>=.5:       # setting threshold to .5
       y_pred2[i]=1
    else:  
       y_pred2[i]=0
from sklearn.metrics import confusion_matrix
cmbl2 = confusion_matrix(y_test, y_pred2)
#Accuracy
from sklearn.metrics import accuracy_score
accuracy2=accuracy_score(y_pred2,y_test)

#Feature importance
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier(n_estimators=250,random_state=420)
model.fit(X,Y)
print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=train.drop('loans_status',axis=1).columns)
feat_importances.nlargest(40).plot(kind='barh')
plt.show()

#Xgboost with cross validation and early stopping
Xg= pd.get_dummies(xtr2,columns = [ 'application_type','home_ownership', 'verification_status','addr_state', 'initial_list_status', 'purpose', 'term2'],drop_first=True)
Xtg= pd.get_dummies(test.drop('loan_status',axis=1),columns = [ 'application_type','home_ownership', 'verification_status','addr_state', 'initial_list_status', 'purpose', 'term2'],drop_first=True)
Xvg= pd.get_dummies(validate.drop('loan_status',axis=1),columns = [ 'application_type','home_ownership', 'verification_status','addr_state', 'initial_list_status', 'purpose', 'term2'],drop_first=True).values
# Get missing columns in the training test
missing_cols = set( Xg.columns ) - set( Xtg.columns )
# Add a missing column in test set with default value equal to 0
for c in missing_cols:
    Xtg[c] = 0
# Ensure the order of column in the test set is in the same order than in train set
Xtg = Xtg[Xg.columns]

import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import cross_validate  #Additional scklearn functions

dtrain = xgb.DMatrix(Xg.values, label=Yy)
dval = xgb.DMatrix(Xvg, label=validate.loan_status.values)
dtest = xgb.DMatrix(Xtg.values)
eval_set = [(Xg, Yy), (Xvg, validate.loan_status.values)]

param = {'max_depth':30, 'silent':1, 'objective':'binary:logistic', 'subsample':0.5,"booster": 'dart', 
         'eval_metric':["error","logloss"],'learning_rate': 0.3, 'seed':420}
#model_cv = xgb.cv(dtrain=dtrain,params= param, nfold=4,metrics=["logloss","error"],num_boost_round=200,early_stopping_rounds=20, seed=420)
model= xgb.train(dtrain=dtrain,params= param,num_boost_round=150,early_stopping_rounds=20,
                        evals= [(dval, 'eval'), (dtrain, 'train')], verbose_eval=10 )
#model.fit(Xg, Yy, early_stopping_rounds=20, eval_metric=["error", "logloss"], eval_set=eval_set, verbose=True)
y_pred_xg = model.predict(dtest)
predictions = [round(value) for value in y_pred_xg]
accuracy_xg2 = accuracy_score(test.loan_status, predictions)
print("Accuracy: %.2f%%" % (accuracy_xg2 * 100.0))

cmxg = confusion_matrix(test.loan_status, predictions)



























g








































rid_search.fit(Xg, Yy )



