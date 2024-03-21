import tensorflow as tf
import os
import cv2
import imghdr
from matplotlib import pyplot as plt
import numpy as np

# prevents out of memory errors
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

data_dir = 'cat data'
image_exts = ['jpeg', 'jpg', 'bmp', 'png']

for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try:
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts:
                print('Image is not in the ext list {}'.format(image_path))
                os.remove(image_path)
        except Exception as e:
            print('Issue with image {]'.format(image_path))

#loading the data using batches of 16
data = tf.keras.utils.image_dataset_from_directory('cat data', batch_size=16, image_size=(256, 256))
data_it = data.as_numpy_iterator()
batch = data_it.next()
# batch[0].shape - returns (16, 128, 128, 3)

# preprocessing the data
data = data.map(lambda x,y: (x/255, y)) # scaling the data batches by dividing by 255 to make as small as possible
scaled_iterator = data.as_numpy_iterator().next() #gives us access to the iterator, same thing as batch above

#fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
#for idx, img in enumerate(batch[0][:4]):
#    ax[idx].imshow(img)
#    ax[idx].title.set_text(batch[1][idx])

train_size = int(len(data) * .7)
val_size = int(len(data) * .2) + 1
test_size = int(len(data) * .1) + 1

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size + val_size).take(test_size)

#actually building the CNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

model = Sequential()
model.add(Conv2D(16, (3, 3), 1, activation='relu', input_shape=(256, 256, 3)))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3, 3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16, (3, 3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(256, activation='relu')) # 256 neurons
model.add(Dense(1, activation='sigmoid'))

model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
# model.summary() - 3696625 values - neural network model

#training the CNN - final results loss = 0.0063, accuracy = 1.0000, val_loss = 0.0955, val_accuracy = 0.9750
hist = model.fit(train, epochs=20, validation_data=val)

from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy

pre = Precision()
re = Recall()
acc = BinaryAccuracy()

for batch in test.as_numpy_iterator():
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)

# test the code with a new image
img = cv2.imread('cat.jpg')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
resize = tf.image.resize(img, (256, 256))
plt.imshow(resize.numpy().astype(int))

yhat = model.predict(np.expand_dims(resize/255, 0))

print(yhat) # yhat = [[0.14706366]]

if yhat > 0.5:
    print(f'The image shows a cat.')
else:
    print(f'The image does not show a cat.')

img = cv2.imread('dog.jpg')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
resize = tf.image.resize(img, (256, 256))
plt.imshow(resize.numpy().astype(int))

yhat = model.predict(np.expand_dims(resize/255, 0))

print(yhat) # yhat = [[0.999999]]

if yhat > 0.5:
    print(f'The image shows a cat.')
else:
    print(f'The image does not show a cat.')

#plotting the performance
fig = plt.figure()
plt.plot(hist.history['loss'], color='red', label='loss')
plt.plot(hist.history['val_loss'], color='black', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc='upper left')
plt.show()

#from tensorflow.keras.models import load_model

#model.save(os.path.join('models', 'acnetestmodel.h5'))
#new_model = load_model(os.path.join('models', 'acnetestmodel.h5'))
#yhatnew = new_model.predict(np.expand_dims(resize/255, 0))

#if yhatnew > 0.5:
#    print("The image shows acne.")
#else:
#    print("The image shows clear skin.")
