# import the necessary packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from PIL import Image
from imutils import paths
import numpy as np
import argparse
import os
from skimage import io
from skimage.transform import resize
from skimage import color
from skimage.feature import hog
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import math
from random import randint

# extract_hog function to extract hog features
def extract_hog(image):
	image = resize(image, (40,40)) 
	features, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualize=True, multichannel=True)
	return list(features)

# show_test_images function to show test images with labes in a frame
def show_test_images(testImagesPaths):
	predictedImagesPaths = []
	predictedImagesLabels = []
	actualImagesLabels = []
	
	for imagePath in testImagesPaths:
		image = io.imread(imagePath)
		label = imagePath.split(os.path.sep)[-2]
		
		features = extract_hog(image)
		features = np.array(features)
		features = features.reshape(1,features.shape[0])
		prediction = model.predict(features)
		
		predictedImagesPaths.append(imagePath)
		predictedImagesLabels.append(le.inverse_transform(prediction)[0])
		actualImagesLabels.append(label)
	
	plt.figure(figsize=(10, 6))
	NoOfImages=10;

	for tt in range(0,NoOfImages):
	    plt.subplot(math.ceil(NoOfImages)/4,5,tt+1)
	    plt.xticks([])
	    plt.yticks([])
	    index = randint(0, len(predictedImagesPaths)-1)
	    image_path = predictedImagesPaths[index]
	    label = predictedImagesLabels[index]
	    alabel = actualImagesLabels[index]
	    img = mpimg.imread(image_path)
	    plt.imshow(img)
	    plt.title("Result = "+label+"/"+alabel) 
	
	plt.show()
	plt.subplots_adjust(wspace=0)

# extract_color_stats function
def extract_color_stats(image):
	# split the input image into its respective RGB color channels
	# and then create a feature vector with 6 values: the mean and
	# standard deviation for each of the 3 channels, respectively
	(R, G, B) = image.split()
	features = [np.mean(R), np.mean(G), np.mean(B), np.std(R),
		np.std(G), np.std(B)]

	return features

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, default="Karolinska Directed Emotional Faces (KDEF)/Train Data",
	help="path to directory containing the facial expression dataset")
ap.add_argument("-m", "--model", type=str, default="svm",
	help="type of python machine learning model to use")
args = vars(ap.parse_args())

# define the dictionary of models our script can use, where the key
# to the dictionary is the name of the model (supplied via command
# line argument) and the value is the model itself
models = {
	"knn": KNeighborsClassifier(n_neighbors=1),
	"naive_bayes": GaussianNB(),
	"logit": LogisticRegression(solver="lbfgs", multi_class="auto"),
	"svm": SVC(kernel="linear"),
	"decision_tree": DecisionTreeClassifier(),
	"random_forest": RandomForestClassifier(n_estimators=100),
	"mlp": MLPClassifier()
}

# grab all image paths in the input dataset directory, initialize our
# list of extracted features and corresponding labels
print("[INFO] Extracting image hog features...")
imagePaths = paths.list_images(args["dataset"])
data = []
labels = []

# loop over our input images
for imagePath in imagePaths:
	# load the input image from disk, compute color channel
	# statistics, and then update our data list
	
	# to extract color features
	#image = Image.open(imagePath)
	#features = extract_color_stats(image)
	
	# to extract hog features
	image = io.imread(imagePath)
	features = extract_hog(image)
	data.append(features)

	# extract the class label from the file path and update the
	# labels list
	label = imagePath.split(os.path.sep)[-2]
	labels.append(label)

# encode the labels, converting them from strings to integers
le = LabelEncoder()
labels = le.fit_transform(labels)

# perform a training and testing split, using 85% of the data for
# training and 15% for evaluation
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.15)

# train the model
print("[INFO] '{}' Model is training...".format(args["model"]))
model = models[args["model"]]
model.fit(trainX, trainY)

# make predictions on our data and show a classification report
print("[INFO] Evaluating...")
predictions = model.predict(testX)

testImagesPaths = paths.list_images('Karolinska Directed Emotional Faces (KDEF)/Test Data')
show_test_images(testImagesPaths)

print(classification_report(testY, predictions, target_names=le.classes_))





