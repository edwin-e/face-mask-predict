from os import listdir
from numpy import asarray
from numpy import save
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
folder = 'with_mask/' 
photos, labels = list(), list()
for file in listdir(folder):
	output = 1
	if file.startswith('without'):
		output = 0
	photo = load_img(folder + file, target_size=(200, 200))
	photo = img_to_array(photo)
	photos.append(photo)
	labels.append(output)
photos = asarray(photos)
labels = asarray(labels)
print(photos.shape, labels.shape)
save('mask_photos.npy', photos)
save('mask_labels.npy', labels)


from os import listdir
from numpy import asarray
from numpy import save
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
folder = 'with_mask1/' 
photos, labels = list(), list()
for file in listdir(folder):
	output = 1
	if file.startswith('without'):
		output = 0
	photo = load_img(folder + file, target_size=(200, 200))
	photo = img_to_array(photo)
	photos.append(photo)
	labels.append(output)
photos = asarray(photos)
labels = asarray(labels)
print(photos.shape, labels.shape)
save('mask_photos1.npy', photos)
save('mask_labels1.npy', labels)


import numpy as np
train_x = np.load('mask_photos.npy')
train_y = np.load('mask_labels.npy')
test_x = np.load('mask_photos1.npy')
test_y = np.load('mask_labels1.npy')


train_x = train_x.reshape(7053,200*200*3)
test_x = test_x.reshape(26,200*200*3)


from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=100)
mlp.fit(train_x ,train_y)
predict_train = mlp.predict(train_x)
predict_test = mlp.predict(test_x)


from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(train_y,predict_train))
print(classification_report(train_y,predict_train))

print(confusion_matrix(test_y,predict_test))
print(classification_report(test_y,predict_test))

import pickle
pickle.dump(mlp, open("mlp.pkl", "wb"))
