import cv2
import numpy as np

face_data=[]
count=0
user_name=input('enter your name:')

#initialise the webcam
cam=cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
while True:
	ret,frame=cam.read()   #reading the frames

	if ret==False:
		print('An error occured')
		continue

	faces=face_cascade.detectMultiScale(frame,1.2,5)    #gives a list of tuples containg x,y,width,height
	faces=sorted(faces,key=lambda f:f[2]*f[3])      #sorting faces on the basis of area=width*height
	for (x,y,w,h) in faces[-1:]:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)        #makes a rectangle around the largest face
	
		
		 #extracting face section from frame: region of interest
		face_section=frame[y-10:y+h+10,x-10:x+w+10]      
		face_section=cv2.resize(face_section,(100,100))   

#saving every 10th face section
		if count%10==0:
			face_data.append(face_section)
			print("number of faces captured:%s"%len(face_data))
		count+=1
	
	cv2.imshow('video frames',frame)
	cv2.imshow('faces',face_section)
	
	key_pressed=cv2.waitKey(1)& 0xFF    #checks every 1ms whether a key is pressed  # 0xFF=masks all the bits expect last 8 bits
	if key_pressed==ord('q'):
		break
#saving the facedata as numpy array and flattening it
face_data=np.array(face_data)
face_data=face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)
# saving in a file
np.save('data/'+user_name+'.npy',face_data)

cam.release()
cv2.destroyAllWindows()