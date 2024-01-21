
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D,Dropout
import cv2
from sklearn.model_selection import train_test_split
import pickle
import os
import pandas as pd
import random
from keras.preprocessing.image import ImageDataGenerator
 
 
################# Parametreler #####################
 
path = "myData" # tüm sınıf klasörlerini içeren klasör
labelFile = 'Train.csv' # tüm sınıf adlarını içeren dosya
batch_size_val=50  # kaç tane birlikte işlenecek
steps_per_epoch_val=2000
epochs_val=10
imageDimesions = (32,32,3)
testRatio = 0.2    # 1000 görüntü bölünürse test için 200 görüntü olacaktır
validationRatio = 0.2 # 1000 görüntü varsa kalan 800'ün %20'si doğrulama için 160 olacaktır
###################################################
 



############################### Görüntülerin İçe Aktarılması ################

count = 0
images = []
classNo = []
myList = os.listdir(path)
print("Total Classes Detected:",len(myList))
noOfClasses=len(myList)
print("Importing Classes.....")
for x in range (0,len(myList)):
    myPicList = os.listdir(path+"/"+str(count))
    for y in myPicList:
        curImg = cv2.imread(path+"/"+str(count)+"/"+y)
        images.append(curImg)
        classNo.append(count)
    print(count, end =" ")
    count +=1
print(" ")
images = np.array(images)
classNo = np.array(classNo)
 
############################### Verileri Böl ####################

X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validationRatio)
 
# X_train = ARRAY OF IMAGES TO TRAIN
# y_train = CORRESPONDING CLASS ID
 
 
############################### HER VERİ SETİ İÇİN GÖRÜNTÜ SAYISINI ETİKET SAYISIYLA EŞLEŞTİRİP EŞLENDİĞİNİ KONTROL ETMEK İÇİN #####################

print("Data Shapes")
print("Train",end = "");print(X_train.shape,y_train.shape)
print("Validation",end = "");print(X_validation.shape,y_validation.shape)
print("Test",end = "");print(X_test.shape,y_test.shape)
assert(X_train.shape[0]==y_train.shape[0]), "The number of images in not equal to the number of lables in training set"
assert(X_validation.shape[0]==y_validation.shape[0]), "The number of images in not equal to the number of lables in validation set"
assert(X_test.shape[0]==y_test.shape[0]), "The number of images in not equal to the number of lables in test set"
assert(X_train.shape[1:]==(imageDimesions))," The dimesions of the Training images are wrong "
assert(X_validation.shape[1:]==(imageDimesions))," The dimesionas of the Validation images are wrong "
assert(X_test.shape[1:]==(imageDimesions))," The dimesionas of the Test images are wrong"
 
 

############################### TÜM SINIFLARA AİT BAZI ÖRNEK GÖRÜNTÜLERİ GÖSTERİN ######################

batch_size_val=30
steps_per_epoch_val=500
epochs_val=40

data=pd.read_csv(labelFile)
print("data shape ",data.shape,type(data))
num_of_samples = []
cols = 3
num_classes = noOfClasses
fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(30, 300))
fig.tight_layout()
for i in range(cols):
    for j,row in data.iterrows():
        x_selected = X_train[y_train == j]
        if len(x_selected)==0:
            continue
        axs[j][i].imshow(x_selected[random.randint(0, len(x_selected)-1), :, :], cmap=plt.get_cmap("gray"))
        axs[j][i].axis("off")
        if i == 2:
            axs[j][i].set_title(str(j)+ "-"+str(row["ClassId"]))
            num_of_samples.append(len(x_selected))

print(num_of_samples)
plt.figure(figsize=(12, 4))
plt.bar(range(0, num_classes), num_of_samples)
plt.title("Distribution of the training dataset")
plt.xlabel("Class number")
plt.ylabel("Number of images")
plt.show()

############################### GÖRÜNTÜLERİN ÖN İŞLENMESİ #############################
 
def grayscale(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img
def equalize(img):
    img =cv2.equalizeHist(img)
    return img
def preprocessing(img):
    img = grayscale(img)     # GRİ TONLAMAYA DÖNÜŞTÜR
    img = equalize(img)      # BİR GÖRÜNTÜDE IŞIKLAMAYI STANDARTLAŞTIRIN
    img = img/255            # 0 İLA 255 YERİNE 0 İLE 1 ARASINDAKİ DEĞERLERİ NORMALLEŞTİRMEK İÇİN
    return img
 
X_train=np.array(list(map(preprocessing,X_train)))  # TÜM GÖRÜNTÜLERİ İRETLEMEK VE ÖN İŞLEMEK İÇİN
X_validation=np.array(list(map(preprocessing,X_validation)))
X_test=np.array(list(map(preprocessing,X_test)))
cv2.imshow("GrayScale Images",X_train[random.randint(0,len(X_train)-1)]) # EĞİTİMİN DOĞRU YAPILDIĞINI KONTROL ETMEK İÇİN
 
############################### 1 DERİNLİK EKLE ###############################

X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_validation=X_validation.reshape(X_validation.shape[0],X_validation.shape[1],X_validation.shape[2],1)
X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)
 
 
############################### GÖRÜNTÜLERİN ARTTIRILMASI: DAHA GENEL HALE GETİRMEK İÇİN ###############################

dataGen= ImageDataGenerator(width_shift_range=0.1,   # 0,1 = %10    1'DEN FAZLA İSE ÖRN. 10 O ZAMAN NO. PİKSEL ÖR. 10 PİKSEL
                            height_shift_range=0.1,
                            zoom_range=0.2,  # 0,2 ORTALAMA 0,8'DEN 1,2'YE ÇIKABİLİR
                            shear_range=0.1,  # KESME AÇISI BÜYÜKLÜĞÜ
                            rotation_range=10)  # DEGREES
dataGen.fit(X_train)
batches= dataGen.flow(X_train,y_train,batch_size=20)  # GÖRÜNTÜLER OLUŞTURMAK İÇİN VERİ ÜRETİCİSİNİN TALEP EDİLMESİ GRUP BOYUTU = NO. HER ÇAĞRILDIĞINDA OLUŞAN GÖRÜNTÜ SAYISI
X_batch,y_batch = next(batches)
 
# GENİŞLETİLMİŞ GÖRÜNTÜ ÖRNEKLERİNİ GÖSTERMEK İÇİN
fig,axs=plt.subplots(1,15,figsize=(20,5))
fig.tight_layout()
 
for i in range(15):
    axs[i].imshow(X_batch[i].reshape(imageDimesions[0],imageDimesions[1]))
    axs[i].axis('off')
plt.show()
 
 
y_train = to_categorical(y_train,noOfClasses)
y_validation = to_categorical(y_validation,noOfClasses)
y_test = to_categorical(y_test,noOfClasses)
 
############################### KONVOLUSYON SİNİR AĞI MODELİ ###############################
def myModel():
    no_Of_Filters=60
    size_of_Filter=(5,5) # BU, ÖZELLİKLERİ ELDE ETMEK İÇİN GÖRÜNTÜ ETRAFINDA HAREKET EDEN ÇEKİRDEKTİR.
                         # BU, 32 32 GÖRÜNTÜ KULLANILDIĞINDA HER SINIRDAN 2 PİKSEL KALDIRILIR
    size_of_Filter2=(3,3)
    size_of_pool=(2,2)  # AŞIRI UYUMU AZALTMAK İÇİN DAHA FAZLA GERNALİLEŞTİRMEK İÇİN TÜM ÖZELLİK HARİTALARININ ÖLÇÜTÜNÜ AZALTIN
    no_Of_Nodes = 500   # NO. GİZLİ KATMANLARDAKİ DÜĞÜMLER
    model= Sequential()
    model.add((Conv2D(no_Of_Filters,size_of_Filter,input_shape=(imageDimesions[0],imageDimesions[1],1),activation='relu')))  # ADDING MORE CONVOLUTION LAYERS = LESS FEATURES BUT CAN CAUSE ACCURACY TO INCREASE
    model.add((Conv2D(no_Of_Filters, size_of_Filter, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool)) # FİLTRE DERİNLİĞİ/SAYISINI ETKİLEMEZ
 
    model.add((Conv2D(no_Of_Filters//2, size_of_Filter2,activation='relu')))
    model.add((Conv2D(no_Of_Filters // 2, size_of_Filter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool))
    model.add(Dropout(0.5))
 
    model.add(Flatten())
    model.add(Dense(no_Of_Nodes,activation='relu'))
    model.add(Dropout(0.5)) # HER GÜNCELLEMEDE DÜŞECEK GİRİŞ DÜĞÜMLERİ 1 HEPSİ 0 YOK
    model.add(Dense(noOfClasses,activation='softmax')) # ÇIKTI KATMANI
    # COMPILE MODEL
    model.compile(Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
    return model
 
 
############################### TRAIN ###############################
model = myModel()
print(model.summary())
history=model.fit_generator(dataGen.flow(X_train,y_train,batch_size=batch_size_val),steps_per_epoch=steps_per_epoch_val,
                            epochs=epochs_val,validation_data=(X_validation,y_validation),shuffle=1)
 
############################### PLOT ###############################
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training','validation'])
plt.title('loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training','validation'])
plt.title('Acurracy')
plt.xlabel('epoch')
plt.show()
score =model.evaluate(X_test,y_test,verbose=0)
print('Test Score:',score[0])
print('Test Accuracy:',score[1])
 
 
# STORE THE MODEL AS A PICKLE OBJECT
pickle_out= open("model_trained.pt","wb")  # wb = WRITE BYTE
pickle.dump(model,pickle_out)
pickle_out.close()
cv2.waitKey(0)

#
#model.save("my_model")
#model.save_weights("weights.h5")