from sklearn.externals import joblib
from skimage.feature import hog
from skimage import feature
import numpy as np
import cv2

wid = 100
dim=(wid,wid)
'''
# load and prepare the image
def load_image(filename):
	# load the image
	img = load_img(filename, grayscale=True, target_size=(28, 28))
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 1 channel
	img = img.reshape(1, 28, 28, 1)
	# prepare pixel data
	img = img.astype('float32')
	img = img / 255.0
	return img
'''
# load an image and predict the class
def run_example():
	# load the image
	#img = load_image('cat.jpg')
	img = cv2.imread("1.jpg")
	imag = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	imge = cv2.resize(imag, (100,100), interpolation=cv2.INTER_AREA)
	imgFeatures, hog_image = hog(imge, orientations=8, pixels_per_cell=(8, 8),
                    cells_per_block=(1, 1), visualise=True)
	lbpF = feature.local_binary_pattern(imge, 10, 5, method="uniform")
	lbpF = lbpF.ravel()
	
	final_vector = np.concatenate((imgFeatures,lbpF))
	fts = final_vector.reshape(1,-1)
	# load Model
	
	model=joblib.load('KNN train model.pkl')#model name
	# predict the class
	digit = model.predict(fts)
	print(digit, ' ---- pridiction')
	#cv2.imshow(str(digit), imag)
	#cv2.waitKey(0)
	#destroyAllWindow();
### Pass the label to image
	if digit[0] == 'Normal':
		print('Normal  <-------pridiction');
		font=cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(img,('Normal'),(50,200), font, 1,(0,255,0),2,cv2.LINE_AA)
		cv2.imshow('Image',img)
		#cv2.imshow(str(digit),imag)
		cv2.waitKey(0)
		cv2.imwrite('Normal.jpg',img)
	else:
		print('Tumor  <-------pridiction');
		font=cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(img,('Tumor'),(50,100), font,1,(0,0,255),2,cv2.LINE_AA)
		cv2.imshow('Image',img)
		#cv2.imshow(str(digit),imag)
		cv2.waitKey(0)
		cv2.imwrite('cancer.jpg',img)

# entry point, run the example
run_example()
