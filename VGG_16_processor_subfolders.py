
# import the necessary packages
# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report,confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler,ModelCheckpoint,TensorBoard
from keras.optimizers import RMSprop,SGD,Adam
from keras.applications import ResNet50,VGG16
from keras.layers import Input, Dropout,Flatten,Dense,GlobalAveragePooling2D
from keras.models import Model,load_model
from keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils
from imutils import paths
from random import shuffle
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import h5py
import cv2

#Needed for top layers of model
from keras.layers.normalization import BatchNormalization
from keras import regularizers

#Adds top classification layers to transfer learning model
def top_layers(baseModel, classes, D):
	# initialize the head model that will be placed on top of
	# the base, then add a FC layer
	l2_constant=0.02
	first_dropout_rate=0.5
	dropout_rate=0.5
	headModel = baseModel.output
	headModel = Flatten()(headModel)
	headModel = Dense(D, activation="relu",kernel_regularizer=regularizers.l2(l2_constant))(headModel)
	headModel = BatchNormalization()(headModel)
	headModel = Dropout(first_dropout_rate)(headModel)
	headModel = Dense(D, activation="relu",kernel_regularizer=regularizers.l2(l2_constant))(headModel)
	headModel = BatchNormalization()(headModel)
	headModel = Dropout(dropout_rate)(headModel)
	headModel = Dense(D, activation="relu",kernel_regularizer=regularizers.l2(l2_constant))(headModel)
	headModel = BatchNormalization()(headModel)
	headModel = Dropout(dropout_rate)(headModel)

	#softmax classification layer
	headModel = Dense(classes, activation="softmax")(headModel)

	# return the model
	return headModel


#modifies learning rate based on epoch
def step_decay(epoch):
    initAlpha=.0020
    if epoch <= 20:
        alpha=initAlpha
    elif epoch <= 40:
        alpha=initAlpha*0.75
    elif epoch <= 60:
        alpha=initAlpha*0.50
    elif epoch <= 80:
        alpha=initAlpha*0.25
    elif epoch <=100:
        alpha=initAlpha*0.175
    elif epoch <= 125:
        alpha=initAlpha*0.10
    else:
        alpha=initAlpha*0.05
            
    return float(alpha)

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

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--set", required=True,
	help="integer index of sherd train/test set")
#Model number (to differentiate different runs
ap.add_argument("-r", "--run", required=True,
	help="run number")
ap.add_argument("-d","--dir",required=True, help="set directory")
args = vars(ap.parse_args())


#name of output model
set_number=str(args["set"])
run_number=str(args["run"])
set_directory=str(args["dir"])
model_id="VGG16_"+set_number+"_"+run_number
print(model_id)

#Define directories to use based on set number for loading data, saving data
train_dataset=set_directory + "\Set_" + set_number +"\Train"
test_dataset=set_directory + "\Set_" + set_number +"\Test"
models_dir=set_directory + "\Set_" + set_number +"\models"
print(train_dataset)
print(test_dataset)


#Parameters for ImageDataGenerator
shift=0.0
zoom=0.3

# construct the image generator for data augmentation
#fill_mode is value put into empty spaces created by rotation or zooming; cval=1.0 means white
aug = ImageDataGenerator(rotation_range=180,
	horizontal_flip=False, vertical_flip=False, width_shift_range=shift, height_shift_range=shift, zoom_range=zoom, fill_mode="constant",cval=1.0)

# determine list of image names,  then extract the class label names from the image paths
print("[INFO] loading images...")
imagePaths_train = list(paths.list_images(train_dataset))
#Randomize order of images in training set
shuffle(imagePaths_train)
imagePaths_test = list(paths.list_images(test_dataset))
#imagePaths_val=list(paths.list_images(val_dataset))
classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths_train]
classNames = [str(x) for x in np.unique(classNames)]
classNames=['Kanaa',  'Wepo', 'Black_Mesa', 'Sosi', 'Dogoszhi', 'Flagstaff', 'Tusayan', 'Kayenta']

(train_data, train_labels) = create_data_arrays(imagePaths_train, verbose=250)
train_data = train_data.astype("float")
train_data = imagenet_utils.preprocess_input(train_data, mode='tf')
print("Training data loaded")

#Load test data separately
(test_data, test_labels) = create_data_arrays(imagePaths_test, verbose=250)
test_data = test_data.astype("float") 
test_data = imagenet_utils.preprocess_input(test_data,mode='tf')
print("Test data loaded")

# set X,Y values to values for train, test
(trainX, testX, trainY, testY) = (train_data,test_data,train_labels,test_labels)


# convert the labels from integers to vectors
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

# load the VGG16 network, head FC layers left off
baseModel = VGG16(weights='imagenet', include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# Set up new head layers
headModel = top_layers(baseModel, len(classNames), 512)

# Set head model as output
model = Model(inputs=baseModel.input, outputs=headModel)

# Freeze top layers for initial training
for layer in baseModel.layers:
	layer.trainable = False

print("VGG16 model loaded")
    

# compile  model after setting base model layers to be non-trainable
print("[INFO] compiling model...")
#Gradient descent optimization algorithm
opt = RMSprop(lr=0.0025)
#Compile, specify loss, optimizer, metrics
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

#Add Learning rate scheduler to model callbacks
callbacks=[LearningRateScheduler(step_decay)]

# Train head only to initialize head weights
print("[INFO] training head...")

epoch_head = 10
model.fit(aug.flow(trainX, trainY, batch_size=32),
	validation_data=(testX, testY), epochs=epoch_head,
	steps_per_epoch=len(trainX) // 32, callbacks=callbacks,verbose=1)

full_model_path = os.path.join(models_dir,'',model_id +'.model')
#Set callbacks for final run, including saving models based on test accuracy (called val_accuracy here),
callbacks=[LearningRateScheduler(step_decay),ModelCheckpoint(full_model_path, monitor='val_accuracy', verbose=1,save_best_only=True,
          save_weights_only=False,mode='max',save_freq='epoch')]

# unfreeze the initial set of deep layers  and make them trainable; remember 0-based count!
for layer in baseModel.layers[0:]:
	layer.trainable = True

#Recompile
print("Re-compiling full model...")
opt = SGD(lr=0.005)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])
print("Final model compiled")

#Train full model
print("Training full model ...")
epoch_full=120
H=model.fit(aug.flow(trainX, trainY, batch_size=32),
	validation_data=(testX, testY), epochs=epoch_full,
	steps_per_epoch=len(trainX) // 32, callbacks=callbacks, verbose=1)

#Load best model for evaluation purposes
model_best = load_model(full_model_path)

#Compute precision/recall, confusion matrix, accuracy/loss graphs; save them in models directory

# evaluate test data
print("[INFO] evaluating test data ...")
predictions = model_best.predict(testX, batch_size=16)
class_report=classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=classNames)
class_file=open(full_model_path  + "_test_class_report.txt","w")
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
np.savetxt(full_model_path  + "_test_confusion_matrix.csv", con_mat, delimiter=",")


# plot the accuracy
plt.style.use("ggplot")
plt.figure(figsize=(11, 8))
plt.plot(np.arange(0, epoch_full), H.history["accuracy"], label="Train accuracy")
plt.plot(np.arange(0, epoch_full), H.history["val_accuracy"], label="Test accuracy")
plt.title(model_id + " Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig(full_model_path + "_accuracy_plot.png")
#plt.show()

# plot the training loss
plt.style.use("ggplot")
plt.figure(figsize=(11, 8))
plt.plot(np.arange(0, epoch_full), H.history["loss"], label="Train loss")
plt.plot(np.arange(0, epoch_full), H.history["val_loss"], label="Test loss")
plt.title(model_id + " Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend()
plt.savefig(full_model_path  + "_loss_plot.png")
#plt.show()

# evaluate train data
print("[INFO] evaluating train data ...")
predictions = model_best.predict(trainX, batch_size=16)
class_report=classification_report(trainY.argmax(axis=1), predictions.argmax(axis=1), target_names=classNames)
class_file=open(full_model_path  + "_train_class_report.txt","w")
class_file.write(class_report)
class_file.close()
print(class_report)
#print(classification_report(np.argmax(testY,axis=1), predictions.argmax(axis=1),target_names=classNames))
print("Confusion matrix")
print(classNames)
con_mat=confusion_matrix(np.argmax(trainY,axis=1), predictions.argmax(axis=1))
#print(confusion_matrix(np.argmax(testY,axis=1), predictions.argmax(axis=1)))
print(con_mat)
#confusion_matrix(np.argmax(testY,axis=1), predictions.argmax(axis=1))
np.savetxt(full_model_path  + "_train_confusion_matrix.csv", con_mat, delimiter=",")