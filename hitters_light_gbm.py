import warnings
import pandas as pd
# !pip install catboost
from catboost import CatBoostRegressor
# !pip install lightgbm
# conda install -c conda-forge lightgbm
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
# !pip install xgboost
from xgboost import XGBRegressor
import numpy as np

"""
Baseball Projesi


Verisetindeki değişkenler
AtBat : 1986 yılında vuruş sayısı

Hits : 1986'daki isabet sayısı

HmRun : 1986 yılında yapılan home run sayısı

Runs : 1986 yılında yapılan koşu sayısı

RBI : 1986 yılında vuruş yapılan koşu sayısı (RBI = Bir vurucunun vuruş yaptığında kaç tane oyuncuya run yaptırdığı) #rbi, run dan daha önemli

Walks : 1986 yılında yürüyüş sayısı

Years : Büyük liglerdeki yıl sayısı

CAtBat : Kariyeri boyunca vuruş sayısı

CHits : Kariyeri boyunca isabet sayısı

CHmRun : Kariyeri boyunca home run sayısı

CRuns : Kariyeri boyunca koşu sayısı

CRBI : Kariyeri boyunca vuruş yapılan koşu sayısı

CWalks : Kariyeri boyunca yürüyüş sayısı

League : Oyuncunun ligini gösteren "A" ve "N" seviyelerine sahip bir faktör

Division : 1986 sonunda oyuncunun bölünmesini gösteren "E" ve "W" seviyelerine sahip bir faktör

PutOuts : 1986 yılında yapılan itiraz sayısı

Assists : 1986 yılında asist sayısı

Errors : 1986'daki hata sayısı (topu yanlış yere atma, topu elinden kaçırma..)

Salary : 1987 açılış gününde yıllık binlerce dolar maaş

NewLeague : 1987'nin başında seviyeleri A ve N oyuncunun ligini gösteren bir faktör

"""



import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from helpers.data_prep import *
from helpers.eda import *
pd.pandas.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 170)

df = pd.read_csv("hitters.csv")
df.head()
df.shape
df["Salary"].value_counts()
df.info()
df["Salary"].describe().T
#Missing Value
df.isnull().sum()
#Eksik Verileri Doldurma

df["YearsGroup"] = pd.cut(df.Years,bins=[0,5,10,15,25],labels=['1', '2', '3' ,'4'])
df["YearsGroup"].value_counts()

df.groupby(["League","Division", "YearsGroup"])["Salary"].mean()
df["Salary1"] = df["Salary"]
df["Salary"].fillna(df.groupby(["League","Division","YearsGroup"])["Salary"].transform("mean"),inplace=True)
df.groupby(["League","Division","YearsGroup"])["Salary"].transform("mean")
df.groupby(["League","Division","YearsGroup"])["Salary"].mean()


num_cols = [col for col in df.columns if len(df[col].unique()) > 20 and df[col].dtypes != 'O']

check_outlier(df,num_cols)

####Feature Engineering


# Kariyeri boyunca yaptığı isabetli atış sayısı / Kariyeri boyunca yaptığı  vuruş sayısı
df['NewHitRate']=df["CHits"]/df["CAtBat"]
df['NewHitRate'].min()

# Kariyeri boyunca yaptığı vuruş sayısı / Yıl
df['NewAtBat']=df["CAtBat"]/df["Years"]

# Yil Assist ve Kosu icin alt siniflar olusturma

df['New_Year'] = pd.cut(x=df['Years'], bins=[0, 5, 10,20],labels = ["New","Experienced","Highly Experienced"])
df['New_Assists']=pd.qcut(x= df['Assists'],q=4,labels=["Very Low","Low","Medium","High"])
df['New_Runs']=pd.qcut(x= df['Runs'],q=4,labels=["Very Low","Low","Medium","High"])

import seaborn as sns

df.corr()

#Degiskenler arasindaki korelasyonu inceleme

sns.heatmap(df.corr(), cmap="Blues",xticklabels="auto", yticklabels="auto")
plt.show()

# Label-One Hot Encoding
binary_cols = [col for col in df.columns if len(df[col].unique()) == 2 and df[col].dtypes == 'O']

for col in binary_cols:
    df = label_encoder(df, col)

df = rare_encoder(df, 0.01)

ohe_cols = [col for col in df.columns if 10 >= len(df[col].unique()) > 2]
df = one_hot_encoder(df, ohe_cols,drop_first=True)

df.head()

from sklearn.preprocessing import RobustScaler


for col in [col for col in num_cols if "Salary" not in col]:
    transformer = RobustScaler().fit(df[[col]])
    df[col] = transformer.transform(df[[col]])



y = df["Salary"]
X = df.drop(["Salary"], axis=1)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)



lgb_model = LGBMRegressor().fit(X_train, y_train)
y_pred = lgb_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

#model_tuning

lgb_model = LGBMRegressor()

lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [500, 1000,10000],
               "max_depth": [3, 5, 8],
               "colsample_bytree": [1, 0.8, 0.5]}



lgbm_cv_model = GridSearchCV(lgb_model,
                             lgbm_params,
                             cv=5,
                             n_jobs=-1,
                             verbose=2).fit(X_train, y_train)

lgbm_cv_model.best_params_

lgbm_tuned = LGBMRegressor(**lgbm_cv_model.best_params_).fit(X_train, y_train)
y_pred = lgbm_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred)) # rmse 77.56


