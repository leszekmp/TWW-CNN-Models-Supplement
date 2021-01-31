
# USAGE
# This program takes  test sherds from labeled folders, classifies by averaging results from all models, creates output csv with results, along with p/r report, confusion matrix.

# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report,confusion_matrix
from keras.models import  load_model
from tensorflow.keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from imutils import paths
import numpy as np
import argparse
import os
import glob
import cv2

#Loads in image data, labels from class subfolder
def create_data_arrays(imagePaths, verbose=-1):
	# initialize the list of features and labels
	data = []
	labels = []

	# loop over the input images
	for (i, imagePath) in enumerate(imagePaths):
		# load the image 
		image = cv2.imread(imagePath)

		#Create class label from directory path
		# /path/to/dataset/{class}/{image}.jpg
		label = imagePath.split(os.path.sep)[-2]

		# resize image, convert image to grayscale, then back to original color scheme for ResNet50, VGG16
		#Original interpolation was cv2.INTER_AREA
		image = cv2.cvtColor(cv2.cvtColor(cv2.resize(image, (224, 224),
			interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2GRAY),cv2.COLOR_GRAY2BGR)
		
		#convert image into array format using Keras function
		image=img_to_array(image)

		# append image array data, label to appropriate lists
		data.append(image)
		labels.append(label)

		# show an update every `verbose` images
		if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
			print("[INFO] processed {}/{}".format(i + 1,
				len(imagePaths)))

	# return image array data, labels
	return (np.array(data), np.array(labels))


# In[2]:

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--set", required=True,
	help="integer index of sherd train/test set")
ap.add_argument("-d","--dir",required=True, help="set directory")
args = vars(ap.parse_args())

set_number=str(args["set"])
set_directory=str(args["dir"])


id=set_number
test_dataset=set_directory + "\Set_" + set_number + "\\Test"
path_to_models=set_directory + "\Set_" + set_number + "\\models"
test_set="Set_" + set_number + "_ensemble"
print(test_dataset)

imagePaths_test=[]
classNames=[]
test_labels=[]
test_data=[]

# determines list of image names,  then extract the class label names from the image paths
print("[INFO] loading images...")
imagePaths_test = list(paths.list_images(test_dataset))
classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths_test]
classNames = [str(x) for x in np.unique(classNames)]
classNames=['Kanaa',  'Wepo', 'Black_Mesa', 'Sosi', 'Dogoszhi', 'Flagstaff', 'Tusayan', 'Kayenta']
num_classes = len(classNames)

#Loads test images into proper format for prediction
(test_data, test_labels) = create_data_arrays(imagePaths_test, verbose=250)
test_data = test_data.astype("float") 
test_data = imagenet_utils.preprocess_input(test_data,mode='tf')
print("Test data loaded")

# Load test data into new array with "test" names
(testX,testY) = (test_data,test_labels)

#Null out unneeded data to save memory
test_data=[]

# convert the labels from integers to vectors
testY = LabelBinarizer().fit_transform(testY)
print("Done!")



#Set path to models to be loaded

modelPaths=os.path.sep.join([path_to_models,"*.model"])
modelPaths=list(glob.glob(modelPaths))                   
models=[]

#Load names of models into array called "models"
print("Loading models...")
l=0
for (i,modelPath) in enumerate(modelPaths):
    #print("[Info] loading model {}/{}".format(i+1),len(modelPaths))
    models.append(load_model(modelPath))
    l=l+1
    print("Model " + str(l) + " of " + str(len(modelPaths)) +" loaded.")


#Run predictions for all models, average results
print("[INFO] evaluating predictions...")
predictions=[]
k=0
for model in models:
    predictions.append(model.predict(testX,batch_size=16))
    k=k+1
    print("Predictions for model " + str(k) + " done." )
#print("Original predictions.shape = " + str(np.shape(predictions)))

#Averages predictions from all models
predictions = np.average(predictions,axis=0)


# Classification report, confusion matrix for results

#predictions = model.predict(testX, batch_size=16)
class_report=classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=classNames)
class_file=open(id + "_" + test_set + "_class_report.txt","w")
class_file.write(class_report)
class_file.close()
print(class_report)
#print(classification_report(np.argmax(testY,axis=1), predictions.argmax(axis=1),target_names=classNames))
print("Confusion matrix")
print(classNames)
con_mat=confusion_matrix(np.argmax(testY,axis=1), predictions.argmax(axis=1))
#print(confusion_matrix(np.argmax(testY,axis=1), predictions.argmax(axis=1)))
print(con_mat)
#confusion_matrix(np.argmax(testY,axis=1), predictions.argmax(axis=1))
np.savetxt(id + "_" + test_set + "_confusion_matrix.csv", con_mat, delimiter=",")

#Section below exports predictions, probabilities as CSV file
label_pred=np.argmax(predictions,axis=1)
label_consensus=np.argmax(testY,axis=1)
Label_list=np.asarray((predictions))

#Write csv file with full info for every image, including probabilities

class_csv=open(id + "_" + test_set + "_classification.csv","w")

j=0

class_csv.write("image_file,consensus,predicted,Kanaa,Wepo,Black_Mesa,Sosi,Dogoszhi,Flagstaff,Tusayan,Kayenta" + '\n')

for i in label_pred:
    #myFormattedList=['%.2f' % elem for elem in predictions[j,]]
    predictions_string=""
    for k in range(num_classes):
        predictions_string=predictions_string+"{:2.3f}".format(predictions[j,k])
        if k < num_classes:
            predictions_string=predictions_string+','
    
    #print(str(predictions[j,]))
    image_id=os.path.basename(imagePaths_test[j])
    #print(image_id)
    #print(classNames[i])
    #print(predictions_string)
    #print(myFormattedList)
    class_csv.write(image_id +',' + classNames[label_consensus[j]] + "," + classNames[i] + ',' + predictions_string + '\n')
    j=j+1

class_csv.close()
print("File " + test_set + "_classification.csv written.")



