
#Gerekli kütüphanelerin import edilmesi

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
from datetime import date

from pyparsing import col
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler,LabelEncoder,StandardScaler,RobustScaler

#Çıktının daha düzenli okunabilmesi adına yapılan işlemler

pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",None)
pd.set_option("display.float_format",lambda x:"%.3f" %x)
pd.set_option("display.width",500)


#veri setinin okunması ardından ilk 5 gözlemin gösterilmesi

df=pd.read_csv("MachineLearning/Beyzbol_Maas_tahmin/hitters.csv")
df.head()

df.shape

#bütün değişken isimleri küçük harfe çevrildi
df.columns=df.columns.str.lower()

#Null değer kontrolü yapalım
df.isnull().sum()
#->59 tane Salary değişkeninde null değerler görülmekte


df.info()

df.describe().T




def grab_col_names(dataframe,cat_th=10,car_th=20):
    #sınıf sayısı 10 dan az ise sayısal görünen kategoriktir.
    #sınıf sayısı 20 dan fazla ise kategorik görünen sayısal.

    #cat_cols , cat_but_car
    cat_cols=[col for col in dataframe.columns if dataframe[col].dtypes=="O"]

    #sayısal görünümlü kategorik değişken
    num_but_cat=[col for col in dataframe.columns if dataframe[col].nunique()<cat_th and
                 dataframe[col].dtypes!="O"]

    #20den büyük olan kategorik değişken ölçülebilirlik yok
    cat_but_car=[col for col in dataframe.columns if dataframe[col].nunique()>car_th and
                 dataframe[col].dtypes=="O"]

    cat_cols=cat_cols+num_but_cat

    cat_cols=[col for col in cat_cols if col not in cat_but_car]


    #num_cols
    num_cols=[col for col in dataframe.columns if  dataframe[col].dtypes !="O"]
    num_cols=[col for col in num_cols if  col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_car: {len(num_but_cat)}")
    return cat_cols,num_cols,cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)


#Korelasyonun İncelenmesi:
df.corr()

# Veri setinin korelasyon matrisi grafiğini oluşturuyoruz.
f, ax = plt.subplots(figsize= [20,15])
sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax, cmap = "magma" )
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()


#oynadığı lig ve mevkilere göre maaşların doldurulması
df.groupby(['league','division']).agg({'salary':['mean','count']})


#Feature Engineering

#Outliers

def outlier_treshols(dataframe,col_name,q1=0.05,q3=0.95):
    quartile1= dataframe[col_name].quantile(q1)
    quartile3= dataframe[col_name].quantile(q3)
    interquantile_range= quartile3-quartile1
    up_limit=quartile3+ 1.5 * interquantile_range
    low_limit=quartile1-1.5*interquantile_range
    return (low_limit,up_limit)

cat_cols, num_cols, cat_but_car = grab_col_names(df)


#Aykırı değer var mı?
def check_outlier(dataframe,col_name):
    low_limit,up_limit=outlier_treshols(dataframe,col_name)
    if dataframe[(dataframe[col_name]>up_limit)|(dataframe[col_name]<low_limit)].any(axis=None):
        return True
    else:
        return False


for col in num_cols:
    print(col,check_outlier(df,col))

#Aykırı değerleri baskılayalım

def replace_with_treshols(dataframe,variable):
    low_limit,up_limit=outlier_treshols(dataframe,variable)
    dataframe.loc[(dataframe[variable]<low_limit),variable]=low_limit
    dataframe.loc[(dataframe[variable]>up_limit),variable]=up_limit

for col in num_cols:
  replace_with_treshols(df,col)

df.describe().T

df['ab/hr'] = df.apply(lambda x: x.atbat / x.hmrun if x.hmrun != 0 else 0,axis=1)

df['c_ab/ht'] = df.apply(lambda x: x.catbat / x.chmrun if x.chmrun != 0 else 0,axis=1)
#BA - Batting average (also abbreviated AVG) - hits divided by at bats
df['avg'] = df.apply(lambda x: x.hits / x.atbat if x.atbat != 0 else 0,axis=1)

df['cavg'] = df.apply(lambda x: x.chits / x.catbat if x.catbat != 0 else 0,axis=1)


df.corrwith(df['salary']).sort_values(ascending=False)[1:]

corr = df.corr()
corr.style.background_gradient(cmap='coolwarm')

encoder = LabelEncoder()

for col in cat_cols + ['years']:
    df[col] = encoder.fit_transform(df[col])


scaler = StandardScaler()

num_cols = df.describe().columns.drop('salary')

df[num_cols] = pd.DataFrame(scaler.fit_transform(df[num_cols]),columns=num_cols)


# X for variables
# y for target

X= df.drop('salary',axis=1)
y= df.salary
# Splitting a dataset with low observations is not the best choice however it is worth to see the results

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=75)

lm = LinearRegression().fit(X_train,y_train)
# predictions

y_pred = lm.predict(X_test)







