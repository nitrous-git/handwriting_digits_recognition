import keras
from keras_preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten
from keras.layers import Activation, Dropout, MaxPooling2D
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import pygame
from keras_preprocessing import image

num_classes = 10
epochs = 6
batch_size = 32
img_width = 28
img_heigth = 28
nb_train_samples = 2000
nb_validation_samples = 2000

# Train directory
train_data_dir = r'C:\Users\Of Corrupted Vision\Documents\Source Python\CNN mnist dataset\flow from directory version\Dataset Mod\train'
# Validation directory
validation_data_dir = r'C:\Users\Of Corrupted Vision\Documents\Source Python\CNN mnist dataset\flow from directory version\Dataset Mod\test'

if K.image_data_format() == 'channels_first':
    input_shape = (3,img_width,img_heigth)
else:
    input_shape = (img_width,img_heigth,3)

model = Sequential()

model.add(Conv2D(64, kernel_size = 3, input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, kernel_size = 3, input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.summary()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True
)

test_datagen=ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width,img_heigth),
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width,img_heigth),
    class_mode='categorical'
)
model.compile(
    optimizer='adam', 
    loss='categorical_crossentropy', 
    metrics=['accuracy'])

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples/batch_size,
    validation_data=validation_generator,
    epochs=epochs,
    validation_steps=nb_validation_samples/batch_size
)

pygame.init()

width = 100
heigth = 100
screen = pygame.display.set_mode((width,heigth))

draw = False
BLACK = (0,0,0)
WHITE = (255,255,255)

brush_color = WHITE
brush_size = 3

done = False
while not done:
    pygame.time.delay(10)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            savefile_name = r'C:\Users\Of Corrupted Vision\Documents\Source Python\CNN mnist dataset\flow from directory version\Dataset Mod\Pygame Save files\digit.jpg'
            pygame.image.save(screen, savefile_name)
            done = True
        if event.type == pygame.MOUSEBUTTONDOWN:
            draw = True
        if event.type == pygame.MOUSEBUTTONUP:
            draw = False
    
    mouse_pos = pygame.mouse.get_pos()
    if draw == True and mouse_pos[0] > 0:
        pygame.draw.circle(screen,brush_color,mouse_pos,brush_size)

    pygame.display.update()

# Make predictions
img_show = image.load_img(r'C:\Users\Of Corrupted Vision\Documents\Source Python\CNN mnist dataset\flow from directory version\Dataset Mod\Pygame Save files\digit.jpg', target_size = (28,28))
plt.imshow(img_show)
plt.show()

img_predict = image.load_img(r'C:\Users\Of Corrupted Vision\Documents\Source Python\CNN mnist dataset\flow from directory version\Dataset Mod\Pygame Save files\digit.jpg', target_size = (28,28))
img_predict = image.img_to_array(img_predict)
img_predict = np.expand_dims(img_predict,axis=0)

result = model.predict(img_predict)
print(result)
if result[0][1] == 1:
    prediction = 'Number : 1'
else:
    prediction ='other number'
print(prediction)
