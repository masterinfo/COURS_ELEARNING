#packages
import keras

# https://www.kaggle.com/code/rae385/handwriting-recognition-using-keras/notebook
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import pandas as pd

#data processing
mnist_train = pd.read_csv('mnist_train.csv')
mnist_test = pd.read_csv('mnist_test.csv')

train_images = mnist_train.iloc[:, 1:].values
train_labels = mnist_train.iloc[:, :1].values
test_images = mnist_test.iloc[:, 1:].values
test_labels = mnist_test.iloc[:, :1].values

#normalize the data
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')
train_images /= 255
test_images /= 255

print("longueur_  image",len(train_images),len(test_images))

#one hot encoding
train_labels = keras.utils.to_categorical(train_labels, 10)
test_labels = keras.utils.to_categorical(test_labels, 10)

# #network topography
# model = Sequential()
# model.add(Dense(512, activation = 'relu', input_shape=(784,)))
# model.add(Dense(10, activation = 'softmax'))
# model.summary()

# Définition du réseau

# N_classes=10
# model = Sequential()
# model.add(Dense(128, activation='relu', input_shape=(784,)))
# model.add(Dropout(0.2))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(N_classes, activation='softmax'))
# # Réumé
# model.summary()



# model = Sequential()
# model.add(Conv2D(filters = 6, kernel_size = 5, strides = 1, activation = 'tanh',
# input_shape = (28,28,1)))
# model.add(MaxPooling2D(pool_size = 2, strides = 2))
# model.add(Conv2D(filters = 16, kernel_size = 5,strides = 1, activation = 'tanh'))
# model.add(MaxPooling2D(pool_size = 2, strides = 2))
# model.add(Flatten())
# model.add(Dense(units = 120, activation = 'tanh'))
# model.add(Dense(units = 84, activation = 'tanh'))
# model.add(Dense(units = 10, activation = 'softmax'))
#
# model.summary()



# descrition du réseau
N_classes=10
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28,28, 1), data_format="channels_last"))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(N_classes, activation='softmax'))
# Résumé
model.summary()


train_images_conv = train_images.reshape(60000, 28, 28, 1)
test_images_conv = test_images.reshape(10000, 28, 28, 1)

#compiling model and training
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
z = model.fit(train_images_conv, train_labels,
                   batch_size=100,
                   epochs=10,
                   verbose=2,
                   validation_data=(test_images_conv, test_labels))

score = model.evaluate(test_images_conv, test_labels, verbose=0)
print("Test Accuracy: ", score[1]*100, "%")

# x=2
# test_images_conv = test_images.reshape(10000, 28, 28, 1)
#
# imaget = test_images[x].reshape(1, 28, 28, 1)
#
# prediction = model.predict(imaget).argmax()
# print(prediction)
#
# for x in range(500):
#     image = test_images[x].reshape(1, 28, 28, 1)
#     prediction = model.predict(image).argmax()
#     label = test_labels[x].argmax()
#     if (prediction != label):
#         plt.title("Sample %d  Prediction: %d Label: %d" % (x, prediction, label))
#         plt.imshow(image.reshape([28,28]), cmap=plt.get_cmap('gray_r') )
#         plt.show()


# visualize the model working
# def predict_test_sample(x):
#     label = test_labels[x].argmax(axis=0)
#     image = test_images[x].reshape([28,28])
#     test_image = test_images[x].reshape(1, 28, 28, 1)
#     prediction = model.predict(test_image).argmax()
#     plt.title("Sample %d  Prediction: %d Label: %d" % (x, prediction, label))
#     plt.imshow(image, cmap=plt.get_cmap('gray_r'))
#     plt.show()


# for x in range(500):
#     image = test_images[x,:].reshape(1,784)
#     prediction = model.predict(image).argmax()
#     label = test_labels[x].argmax()
#     if (prediction != label):
#         plt.title("Sample %d  Prediction: %d Label: %d" % (x, prediction, label))
#         plt.imshow(image.reshape([28,28]), cmap=plt.get_cmap('gray_r') )
#         plt.show()


nb=len(test_images)
compteurmauvaisLabel=0
for x in range(0,nb):
    # image = test_images[x,:].reshape(1,784)
    image = test_images[x].reshape(1, 28, 28, 1)
    prediction = model.predict(image).argmax()
    label = test_labels[x].argmax()
    if (prediction != label):
        plt.title("Sample %d  Prediction: %d Label: %d" % (x, prediction, label))
        plt.imshow(image.reshape([28,28]), cmap=plt.get_cmap('gray_r') )
        plt.show()
        compteurmauvaisLabel=compteurmauvaisLabel+1

print("-----------------------------------------------------")
print("Nombre de mauvais labels: ", compteurmauvaisLabel)
print("Pourcentage de mauvais labels: ", compteurmauvaisLabel/nb*100, "%")