import cv2

cam=cv2.VideoCapture(0)
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

while True:
	ret,frame=cam.read()

	if ret==False:       #checking whether frame is available
		print('an error occured')
		continue          #it continues implementing the loop untill it gets a frame
	

	faces=face_cascade.detectMultiScale(frame,1.3,5)
	print(faces)
	for (x,y,w,h) in faces:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

	cv2.imshow("my video",frame)

	key_pressed=cv2.waitKey(1)& 0xFF  #checks every 1ms whether a key is pressed  # 0xFF=masks all the bits expect last 8
	if key_pressed==ord('q'):
		break

cam.release()
cv2.destroyAllWindows()