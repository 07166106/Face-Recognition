#Facial Recognition

import cv2
import numpy as np
import os

#######KNN CODE######

def distance(v1,v2):
    return np.sqrt(((v1-v2)**2).sum())

def knn(train,test,k=5):
    dist=[]
    for i in range(train.shape[0]):
        ix=train[i,:-1]
        iy=train[i,-1]
        print(ix,end = '')
        print(iy)
        d=distance(test,ix)
        dist.append([d,iy])
    dk=sorted(dist,key=lambda x:x[0])[:k] 
     #x[0] cz want to sort it based on the distance d and upto K
     #Retrieving only the labels ie. only the last column values which is the label is being retrieved 
    labels=np.array(dk)[:,-1]
    #getting the label frequencies (individually)
    output=np.unique(labels,return_counts=True)
    #Finding maximum frequency along with the corresponding label
    index=np.argmax(output[1])
    return output[0][index]

#initializing web cam
cap=cv2.VideoCapture(0)
#Face detection Classifier
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
#Data preparation 
#Labels for the given file
class_id=0
#mapping Id with name using a dictionary
names={}
labels=[]
face_data=[]
dataset_path="./data/"
#going through each file in the data folder
for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        #creating a mapping between class_ID and name (stores the complete name before '.npy' extension)
        names[class_id]=fx[:-4]
        #loading data
        data_item=np.load(dataset_path+fx)
        face_data.append(data_item)
        #each frame(10,000 columns) will be alloted a label 0 (class_id), all data belonging to the first file
        #creating labels for individual frames
        target=class_id*np.ones((data_item.shape[0],))
        #created a 1-D array for the labels
        class_id+=1
        labels.append(target)
face_dataset=np.concatenate(face_data,axis=0)
face_labels=np.concatenate(labels,axis=0).reshape((-1,1))
train_set=np.concatenate((face_dataset,face_labels),axis=1)
print(train_set.shape)
#testing 
while True:
    ret,frame=cap.read()
    if ret==False:
        continue
    gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray_frame,1.3,5)
    if len(faces)==0:
        continue
    for face in faces:
        x,y,w,h=face
        #extracting or cropping of the region of interest
        offset=10
        face_section=gray_frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section=cv2.resize(face_section,(100,100))
        #prediction
        out=knn(train_set,face_section.flatten())
        #Display the output on the screen
        pred_name=names[int(out)]
        cv2.putText(gray_frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        cv2.rectangle(gray_frame,(x,y),(x+w,y+h),(0,255,255),2)
    cv2.imshow("Gray_Frame",gray_frame)
    key_pressed=cv2.waitKey(1) & 0xFF
    if key_pressed==ord('q'):
        break
cap.release()
cv2.destroyAllWindows
 