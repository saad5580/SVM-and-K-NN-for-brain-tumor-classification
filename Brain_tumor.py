import cv2 # It mainly focuses on image processing, video capture and analysis including features like face detection and object detection.
import os,glob #These modules is used to retrieve files/pathnames
import numpy as np #NumPy is a python library used for working with arrays.
from skimage.feature import hog , local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.svm import SVC
""" The objective of a Linear SVC (Support Vector Classifier) is to fit to the data you provide,
returning a "best fit" hyperplane that divides, or categorizes, your data. From there, after getting the hyperplane,
you can then feed some features to your classifier to see what the "predicted" class is."""

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

path='Dataset/*'

data_path = os.path.join(path,'*g') #os.path.join concatenate all the paths of images intelligently which is contained in path variable.
imagePaths = glob.glob(data_path) #The glob. glob returns the list of files with their full path
print('Total dataset images length:')
print(len(imagePaths))

data = []
LBP = []
labels = []
hogFeatures = []
featurevector = [[0 for x in range(0)] for y in range(0)] #2d array

# loop over the input images

for imagePath in imagePaths:
	# load the image, pre-process it, and store it in the data list
	image = cv2.imread(imagePath,0)
	image = cv2.resize(image, (100,100),interpolation=cv2.INTER_AREA)
	fd, hog_image = hog(image, orientations=8, pixels_per_cell=(8, 8),cells_per_block=(1, 1), visualise=True)
	lbp = local_binary_pattern(image, 10,15,  method= "uniform")
	#hogFeatures.append(fd)
	#LBP.append(lbp)
	llb = lbp.ravel() #ravel, which is used to change a 2-dimensional array or a multi-dimensional array into a contiguous flattened array
	hogFeatures = fd
	features = np.concatenate((hogFeatures,llb))
	featurevector.append(features)
	#extract labels
	#data.append(image)
	l = label = imagePath.split(os.path.sep)[-2].split("_")
	labels.append(l)
	

	
print('\n')
print('********** Dataset labels *********''\n')
print(labels)
print('\n')
print('********** Total hogFeatures *********')
print(len(hogFeatures))
print('\n')

X_train, X_test, y_train, y_test = train_test_split(featurevector, labels, test_size = 0.30)

print('train images ',len(X_train) )
print('train labels ',len(y_train))
print('\n')
print('test images ',len(X_test))
print('test labels',len(y_test))
print('\n')

print('********** Total LBP_Features *********')
print(len(llb))

########## Classifiers #############
##########   SVM       #############
"""
svclassifier = svm.SVC()
svclassifier = SVC(kernel='linear')
svclassifier.probability = True
svclassifier.fit(X_train, y_train)

#Now making prediction
y_pred = svclassifier.predict(X_test)
print('\n')
print("Actual Labels :    "'\n',y_test)
print('\n')
print("Predicted Labels : "'\n',y_pred)
print('\n')
# accuracy 
accuracy = svclassifier.score(X_test, y_test)
print('SVM Accuracy = ',accuracy*100)
print('\n')
"""
##########   KNN       ############

classifier= KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test)

#Now making prediction
y_pred = classifier.predict(X_test)
print('\n')
print("Actual Labels :    ", y_test)
print('\n')
print("Predicted Labels : ",y_pred)

# accuracy 
accuracy = classifier.score(X_test, y_test)
print('KNN Accuracy',accuracy*100)
print('\n')

####################################


# confusion matrix
print('********** Confusion Matrix *********')
print(confusion_matrix(y_test,y_pred))
#print(classification_report(y_test,y_pred))

#joblib.dump(classifier,'KNN train model.pkl')
#print('Model saved...!')

