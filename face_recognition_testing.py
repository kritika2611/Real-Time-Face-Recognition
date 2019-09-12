import cv2
import numpy as np
import os

names={}
path='./data/'
cnt=0
id=0
X=[]
Y=[]
#data preparation
for f in os.listdir(path):
	if f.endswith('.npy'):
		names[id]=f[:-4]
		data=np.load(path+f)
		X.append(data)
	label=id*np.ones((data.shape[0],))
	Y.append(label)
	id+=1
X=np.concatenate(X,axis=0)
Y=np.concatenate(Y,axis=0).reshape((-1,1))
train=np.concatenate((X,Y),axis=1)
print(X.shape)
print(Y.shape)
print(train.shape)
print(names)

#ALGORITHM
def distance(p1,p2):
	return np.sum((p2-p1)**2)**0.5

def knn(X_values,Y_values,test,k=11):
	dist=[]
	for i in range(X_values.shape[0]):
		d=distance(test,X_values[i])
		dist.append([d,Y_values[i]])
	dist=sorted(dist,key=lambda f:f[0])
	dist=np.array(dist)[:,1]
	dist=dist[:k]
	print(dist)

	u=np.unique(dist,return_counts=True)
	pred=u[0][np.argmax(u[1])]
	return pred


#testing
cam=cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

while True:
	ret,frame=cam.read()
	if ret==False:
		continue
	
	faces=face_cascade.detectMultiScale(frame,1.3,5)
	for (x,y,w,h) in faces:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
		face_section=frame[y-10:y+h+10,x-10:x+w+10]
		face_section=cv2.resize(face_section,(100,100))
		pred=knn(X,Y,face_section.flatten(),k=7)
		pred_name=names[int(pred)]
		cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
	cv2.imshow('video',frame)


	key_pressed=cv2.waitKey(1) & 0xFF
	if key_pressed==ord('q'):
		break
	
cam.release()
cv2.destroyAllWindows()

