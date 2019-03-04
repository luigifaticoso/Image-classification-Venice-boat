import pandas
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy
from keras.models import load_model

# # returns a compiled model
# # identical to the previous one
# model = load_model('my_model.h5')

# Sets
seed = 42
numpy.random.seed(seed)

train_datagen = ImageDataGenerator(rescale = 1./255,
shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
train = train_datagen.flow_from_directory('train_sc5',
target_size = (30, 100),
batch_size = 32,
class_mode = 'categorical')
validation = train_datagen.flow_from_directory(
    'train_sc5',
    subset="validation",
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)
num_classes = len(train.class_indices)

test = test_datagen.flow_from_directory('test_folder',
target_size = (30, 100),
batch_size = 32,
has_ext=True,
x_col="filename",
y_col="boat",
class_mode = 'categorical',
shuffle=False)


classifier = Sequential()
# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = train.image_shape, activation = 'relu'))
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.49))
# Step 3 - Flattening
classifier.add(Flatten())
# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))

classifier.add(Dense(units = 64, activation = 'relu'))



#avevo messo sigmoid
classifier.add(Dense(num_classes, activation = 'softmax'))
#avevo messo sparse_categorical_crossentropy
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.summary()

classifier.fit_generator(train,steps_per_epoch=train.samples // batch_size,validation_data=validation,validation_steps=validation.samples // batch_size,callbacks=[ReduceLROnPlateau(), EarlyStopping(monitor='val_loss', patience=ES_patience)],sverbose=1)
#Evaluating
test_steps_per_epoch = numpy.math.ceil(float(test.samples) / test.batch_size)
raw_predictions = model.predict_generator(test, steps=test_steps_per_epoch)
predictions = numpy.argmax(raw_predictions, axis=1)

test_classes=[]
for c in test.classes:
    for n in range(len(class_dict)):
        if class_dict[n][1] == c:
            test_classes.append(n)

print("Prediction:  " + str(numpy.bincount(predictions)))
print("Groundtruth: " + str(numpy.bincount(test_classes)))

print(metrics.classification_report(test_classes, predictions, target_names=class_labels))


