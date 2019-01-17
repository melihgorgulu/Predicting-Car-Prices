# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 12:54:48 2018

@author: melih
"""
######### KUTUPHANELERIN YUKLENMESI##########

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score  #Model karsilastirmasi icin r^2 metodu
import numpy as np
import seaborn as sns

###########################################

############ VERI ON ISLEME ###############

#Verilerin Yuklenmesi#
veriler=pd.read_csv('out.csv',low_memory=False) 
# Burada pandas kutuphanesi bir sutunun tipini(orn int) anlayabilmek icin
# Sutunun sonuna kadar bakmalidir. Bazi sutunlar farkli tipte degisken iceriyor
# memoryi cok kullandigindan oturu hata vermemesi icin low_memory=False kullandik
veriler1 = veriler[(veriler['MotorGucu(hp)'] <= 600)]#Shape of passed values is (15, 27484), indices imply (15, 30911) hatasi almamak icin
veriler1 = veriler1[(veriler1['MotorHacmi(cc)'] <= 6000)]


### xde bazi istenmeyen sutunlarin atilmasi icin dataframe split yapiyoruz
x1=veriler.iloc[:,3:17] # Marka- Hasar Durumu sutunlarini ve bu sutunlarin arasindaki sutunlari aldik
# print(veriler.columns.get_loc("Durumu"))  sutun numarasi alma
x2=veriler.iloc[:,veriler.columns.get_loc("Durumu"):veriler.columns.get_loc("RuhsatKaydi")-2]

x3=veriler.iloc[:,veriler.columns.get_loc("RuhsatKaydi")+1:185]

#eksik veriler
from sklearn.preprocessing import Imputer
imputer= Imputer(missing_values='NaN',strategy='most_frequent',axis=0)  ## strateji olarak 
## null degiskenlerin yerine mod degerini koyuyoruz
X1=x3.iloc[:,0:19].values
X2=x2.iloc[:,0:144].values
imputer=imputer.fit(X1[:,0:19])
X1[:,0:19]=imputer.transform(X1[:,0:19])   #x3deki eksik veriler duzeltildi


imputer2=imputer.fit(X2[:,x2.iloc[:,0:144].columns.get_loc("BoyaliParcalar-SagArkaCamurluk"):144])
X2[:,x2.iloc[:,0:144].columns.get_loc("BoyaliParcalar-SagArkaCamurluk"):144]=imputer2.transform(X2[:,x2.iloc[:,0:144].columns.get_loc("BoyaliParcalar-SagArkaCamurluk"):144]) 
# x2 deki eksik veriler duzeltildi



# duzeltilmis eksik verilerin dataframe donusumu
x3_duzeltilmis = pd.DataFrame(data = X1, index = range(30911), columns=["ESP","DeriDoseme","ElektrikliKoltuklar","IsitmaliKoltuklar","Klima","OtomatikCam","AlasimJant","ElektrikliAynalar","FarSensoru","FarYikamaSistemi","OtomatikKapi","ParkSensoru","SurguluKapiTek","SurguluKapiCift","XenonFar","CekiDemiri","CDCalar","RadioKasetcalar"] )

x2_duzeltilmis= pd.DataFrame(data = X2, index = range(30911), columns=["Durumu","ABC","ABS","EBP","ASR","ESPVSA","Airmatic","EDL","EBA","EBD","TCS","BAS","Distronic","YokusKalkisDestegi","GeceGorus","SerittenAyrilmaIkazi","SeritDegistirmeYardimcisi","HavaYastigiSurucu","HavaYastigiYolcu","HavaYastigiYan","HavaYastigiDiz","HavaYastigiPerde","HavaYastigiTavan","KorNoktaUyariSistemi","LastikArizaGostergesi","YorgunlukTespitSistemi","Isofix","Alarm","MerkeziKilit","Immobilizer","DeriKoltuk","KumasKoltuk","DeriKumasKoltuk","ElektrikliOnCamlar","ElektrikliArkaCamlar","KlimaAnalog","KlimaDijital","OtmKararanDikizAynasi","OnKolDayama","ArkaKolDayama","AnahtarsizGirisveCalistirma","6IleriVites","7IleriVites","HidrolikDireksiyon","FonksiyonelDireksiyon","AyarlanabilirDireksiyon","DeriDireksiyon","AhsapDireksiyon","IsitmaliDireksiyon","KoltuklarElektrikli","KoltuklarHafizali","KoltuklarKatlanir","KoltuklarOnIsitmali","KoltuklarArkaIsitmali","KoltuklarSogutmali","HizSabitleyici","AdaptiveCruiseControl","SogutmaliTorpido","YolBilgisayari","KromKaplama","AhsapKaplama","HeadupDisplay","StartStop","GeriGorusKamerasi","OnGorusKamerasi","3SiraKoltuk","Hardtop","FarLED","FarHalojen","FarXenon","FarBiXenon","FarSis","FarAdaptif","FarGeceSensoru","FarYikama","AynalarElektrikli","AynalarOtomKatlanir","AynalarIsitmali","AynalarHafizali","ParkSensoruArka","ParkSensoruOn","ParkAsistani","AlasimliJant","Sunroof","PanoramikCamTavan","YagmurSensoru","ArkaCamBuzCozucu","PanoramikOnCam","RomorkCekiDemiri","AkilliBagajKapagi","RadyoKasetcalar","RadyoCDCalar","RadyoMP3Calar","TVNavigasyon","BluetoothTelefon","USBAUX","AUX","iPodBaglantisi","6Hoparlor","CDDegistirici","ArkaEglencePaketi","DVDDegistirici","BoyaliParcalar-SagArkaCamurluk","DegisenParcalar-SagArkaCamurluk","BoyaliParcalar-SagOnCamurluk","DegisenParcalar-SagOnCamurluk","ArkaKaput","ArkaTampon","MotorKaputu","SagArkaKapi","SagArkaCamurluk","SagOnKapi","SagOnCamurluk","SolArkaKapi","SolArkaCamurluk","SolOnKapi","SolOnCamurluk","Tavan","OnTampon","BoyaliParcalar-SolOnKapi","BoyaliParcalar-SolOnCamurluk","DegisenParcalar-SolOnKapi","DegisenParcalar-SolOnCamurluk","BoyaliParcalar-MotorKaputu","DegisenParcalar-MotorKaputu","BoyaliParcalar-SagOnKapi","BoyaliParcalar-SagArkaKapi","BoyaliParcalar-SolArkaKapi","BoyaliParcalar-SolArkaCamurluk","BoyaliParcalar-ArkaKaput","BoyaliParcalar-ArkaTampon","DegisenParcalar-SagOnKapi","DegisenParcalar-SagArkaKapi","DegisenParcalar-SolArkaKapi","DegisenParcalar-SolArkaCamurluk","DegisenParcalar-ArkaKaput","DegisenParcalar-ArkaTampon","BoyaliParcalar-OnTampon","DegisenParcalar-OnTampon","BoyaliParcalar-Tavan","DegisenParcalar-Tavan","MenzilKM","Km"] )

#dataframe birlestirme islemi
x4=pd.concat([x1,x2_duzeltilmis],axis=1)  
x_eksikverisiz=pd.concat([x4,x3_duzeltilmis],axis=1) # TEMIZLENMIS VE EKSIK VERILERI HALLEDILEN x(BAGIMSIZ DEGISKEN)

###### x kumesindeki eksik veriler duzeltildi ve bazi istenmeyen sutunlar atildi######

### bu noktadan sonra elimizde y(tahmin etmek istedigimiz fiyatlarin oldugu) dataframe 
### ve istenmeyen sutunlardan ve eksik verilerden arindirilmis features'larimizin
#### oldugu x_eksikverisiz dataframe'i var

#----------------------------------------------------------------------------------------


##### Encoder Kategorik---> Numeric
  ### X_eksizverisi_sol x_eksikverisiz'in encode edilmesi icin ayrilan sol kismi
X_eksizverisiz_sol=x_eksikverisiz.iloc[:,0:15].values
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# Hata almamak icin tek tek fit transform yaptik
X_eksizverisiz_sol[:,0]= le.fit_transform(X_eksizverisiz_sol[:,0])
X_eksizverisiz_sol[:,1]= le.fit_transform(X_eksizverisiz_sol[:,1])
X_eksizverisiz_sol[:,2]= le.fit_transform(X_eksizverisiz_sol[:,2].astype(str))
X_eksizverisiz_sol[:,4]= le.fit_transform(X_eksizverisiz_sol[:,4].astype(str))
X_eksizverisiz_sol[:,5]= le.fit_transform(X_eksizverisiz_sol[:,5].astype(str))
X_eksizverisiz_sol[:,7]= le.fit_transform(X_eksizverisiz_sol[:,7].astype(str))
X_eksizverisiz_sol[:,10]= le.fit_transform(X_eksizverisiz_sol[:,10].astype(str))
X_eksizverisiz_sol[:,11]= le.fit_transform(X_eksizverisiz_sol[:,11].astype(str))
X_eksizverisiz_sol[:,12]= le.fit_transform(X_eksizverisiz_sol[:,12].astype(str))
X_eksizverisiz_sol[:,13]= le.fit_transform(X_eksizverisiz_sol[:,13].astype(str))
X_eksizverisiz_sol[:,14]= le.fit_transform(X_eksizverisiz_sol[:,14].astype(str)) 

X_encoded= pd.DataFrame(data = X_eksizverisiz_sol, index = range(30911), columns=["Marka","Seri","Model","Yil","Yakit","Vites","KM","KasaTipi","MotorGucu(hp)","MotorHacmi(cc)","Cekis","Renk","Garanti","HasarDurumu","Durumu"])

# x_encoded x'in sol kismininin encode edilmis hali ve bir dataframe.
x_eksikverisiz_sag=x_eksikverisiz.iloc[:,15:-1]

x=pd.concat([X_encoded,x_eksikverisiz_sag],axis=1)
x.fillna(1, inplace=True)
plt.scatter(x['MotorGucu(hp)'],x['MotorHacmi(cc)'],color='blue')
plt.title('MotorGucu ve MotorHacmi anomalies ')
plt.xlabel('MotorGucu(hp)')
plt.ylabel('MotorHacmi(cc)')
plt.show()

x = x[(x['MotorGucu(hp)'] <= 600)]
x = x[(x['MotorHacmi(cc)'] <= 6000)]
x_re=x.iloc[:,0:15]
y = veriler1.iloc[:,0:1] #y bagimsiz degiskenlerin alinmasi 

# X DATAFRAME'INDE CESITLI TEKNIKLER ILE DATA FRAME AZALTMA
# corr kullanma 
x_1=pd.concat([y,x_re],axis=1) # hata veriyordu bunun icin y bagimli degiskeni de alindi
corr=x_1.corr()
#birbiri ile corelasyonu 0.9'dan fazla olan sutunlardan birini secen kod
columns = np.full((corr.shape[0],), True, dtype=bool)
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if corr.iloc[i,j] >= 0.9:
            if columns[j]:
                columns[j] = False
selected_columns = x_1.columns[columns]
x_1 = x_1[selected_columns]

# Backward elimination ile p-value'lara gore features eleme
selected_columns = selected_columns[1:].values
import statsmodels.formula.api as sm
def backwardElimination(x, Y, sl, columns):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(Y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
                    columns = np.delete(columns, j)
                    
    regressor_OLS.summary()
    return x, columns
SL = 0.05
data_modeled, selected_columns = backwardElimination(x_1.iloc[:,1:].values, x_1.iloc[:,0].values, SL, selected_columns)

data = pd.DataFrame(data = data_modeled, columns = selected_columns) #p value ve correlation"a gore secilen features'larin alinmasi



# Data dataframe'inin gorsellestirilmesi
#Gorsellestirme
# Bazi Featrueslarin ikili karsilastirmasi
data_temp=pd.concat([y,data],axis=1) #data_temp is for graph
plt.scatter(data_temp['Marka'],data_temp['Fiyat'],color='blue')
plt.title('Price-Brand Relationship')
plt.xlabel('Encoded Brand')
plt.ylabel('Price')
plt.show()

plt.scatter(data_temp['KM'],data_temp['Fiyat'],color='blue')
plt.title('Price-KM Relationship')
plt.xlabel('KM')
plt.ylabel('Price')
plt.show()

plt.scatter(data_temp['Yil'],data_temp['Fiyat'],color='blue')
plt.title('Price-Year Relationship')
plt.xlabel('Year')
plt.ylabel('Price')
plt.show()

##########################################################################################

#### bu noktadan sonra elimizde kirpilmis bagimsiz degisken data ve bagimli degisken y var


#verilerin egitim ve test icin bolunmesi
from sklearn.cross_validation import train_test_split
x_train, x_test,y_train,y_test = train_test_split(data,y,test_size=0.33, random_state=0)

#verilerin olceklenmesi

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
Y_test=sc.fit_transform(y_test)
Y_train=sc.fit_transform(y_train)

## linear regression fit

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,Y_train)



# SVR fit
from sklearn.svm import SVR

# kernel= rbf SVM kullanimi
svr_reg1=SVR(kernel='rbf')
svr_reg1.fit(X_train,Y_train)

# kernel= poly SVM kullanimi
svr_reg2=SVR(kernel='poly')
svr_reg2.fit(X_train,Y_train)


# kernel= linear SVM kullanimi

svr_reg3=SVR(kernel='linear')
svr_reg3.fit(X_train,Y_train)



# Decision Tree Regression fit
from sklearn.tree import DecisionTreeRegressor
r_dt=DecisionTreeRegressor(random_state=0)
r_dt.fit(X_train,Y_train)

# Random Forest Regression fit
from sklearn.ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor(n_estimators=10,random_state=0) 
# n_estimators, kac tane decision tree cizilecegini belirten parametre
rf_reg.fit(X_train,Y_train)


# Ozet R^2 degerleri
print('--------------------------------')
print("Linear Reg r^2 degeri : ")
print(r2_score(Y_test,lr.predict(X_test)))


print("Support Vector Machine r^2 degeri(rbf) : ")
print(r2_score(Y_test,svr_reg1.predict(X_test)))

print("Support Vector Machine r^2 degeri(poly) : ")
print(r2_score(Y_test,svr_reg2.predict(X_test)))

print("Support Vector Machine r^2 degeri(linear) : ")
print(r2_score(Y_test,svr_reg3.predict(X_test)))

print("Decision Tree r^2 degeri : ")
print(r2_score(Y_test,r_dt.predict(X_test)))


print("Random Forest r^2 degeri : ")
print(r2_score(Y_test,(rf_reg.predict(X_test))))
print('--------------------------------')

# Secilen Modele Gore tahmin
tahmin=sc.inverse_transform(rf_reg.predict(X_test))

# Tahminlerin Gorsellestirimesi

# Linear Regression
print('### Tahminlerin Gorsellestirilmesi ###')
print('Bu degerler Lineare ne kadar yakinsa ve(genelde egimi yuksekse basari daha iyi oluyor) o kadar iyi.')
print('--------------------------------------------------------')
plt.scatter(Y_test,lr.predict(X_test),color='red')
plt.title('Linear Regression')
plt.xlabel('Tahmin')
plt.ylabel('Gercek Deger')
plt.show()

# kernel= rbf SVM kullanimi
print('--------------------------------------------------------')
plt.scatter(Y_test,svr_reg1.predict(X_test),color='red')
plt.title('SVR (kernel=rbf) Regression')
plt.xlabel('Tahmin')
plt.ylabel('Gercek Deger')
plt.show()

# kernel= poly SVM kullanimi
print('--------------------------------------------------------')
plt.scatter(Y_test,svr_reg2.predict(X_test),color='red')
plt.title('SVR (kernel=poly) Regression')
plt.xlabel('Tahmin')
plt.ylabel('Gercek Deger')
plt.show()

# kernel= linear SVM kullanimi
print('--------------------------------------------------------')
plt.scatter(Y_test,svr_reg3.predict(X_test),color='red')
plt.title('SVR (kernel=linear) Regression')
plt.xlabel('Tahmin')
plt.ylabel('Gercek Deger')
plt.show()

# Decision Tree Regression fit
print('--------------------------------------------------------')
plt.scatter(Y_test,r_dt.predict(X_test),color='red')
plt.title('Decision Tree Regression')
plt.xlabel('Tahmin')
plt.ylabel('Gercek Deger')
plt.show()

# Random Forest Regression fit
print('--------------------------------------------------------')
plt.scatter(Y_test,rf_reg.predict(X_test),color='red')
plt.title('Random Forest Regression ')
plt.xlabel('Tahmin')
plt.ylabel('Gercek Deger')


y_test=sc.inverse_transform(Y_test)  #scaled verinin eski haline dondurulmesi boylece sayilar anlam ifade ediyor


# Gorsellestirme devami(Grafiklerin ust uste biniyordu ust uste binmemesi icin heatmap'i en son cizdirdik )
# Correlation Matrix Heatmap
f, ax = plt.subplots(figsize=(10, 6))
corr = data.corr()
hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="coolwarm",fmt='.2f',
                 linewidths=.05)
f.subplots_adjust(top=0.93)
t= f.suptitle('Car\'s Features Correlation Heatmap', fontsize=14)



