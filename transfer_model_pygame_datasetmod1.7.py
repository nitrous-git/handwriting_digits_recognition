import keras
import matplotlib.pyplot as plt
from keras.models import load_model
import numpy as np
import pygame
from keras_preprocessing import image


# load trained weights
model = load_model(r'C:\Users\Of Corrupted Vision\Documents\Source Python\CNN mnist dataset\logs\model_1.h5')

# Model compile
model.compile(
    optimizer='adam', 
    loss='categorical_crossentropy', 
    metrics=['accuracy'])

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
    pygame.time.delay(30)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
        if event.type == pygame.MOUSEBUTTONDOWN:
            draw = True
        if event.type == pygame.MOUSEBUTTONUP:
            draw = False
    
    mouse_pos = pygame.mouse.get_pos()
    pressed = pygame.key.get_pressed()
    if draw == True and mouse_pos[0] > 0:
        pygame.draw.circle(screen,brush_color,mouse_pos,brush_size)

    # pres enter to save
    if pressed[pygame.K_RETURN]:
        savefile_name = r'C:\Users\Of Corrupted Vision\Documents\Source Python\CNN mnist dataset\flow from directory version\Dataset Mod\Pygame Save files\digit.jpg'
        pygame.image.save(screen, savefile_name)
        #print('DIGIT SAVED ---WAIT----')  
    if pressed[pygame.K_0]:
        brush_color = BLACK
        brush_size = 20
    if pressed[pygame.K_1]:
        brush_color = WHITE
        brush_size = 3

    pygame.display.update()

    # Make predictions
    #print('MAKING PREDICTION ---WAIT---')
    img_show = image.load_img(r'C:\Users\Of Corrupted Vision\Documents\Source Python\CNN mnist dataset\flow from directory version\Dataset Mod\Pygame Save files\digit.jpg', target_size = (28,28))
    plt.imshow(img_show)
    #plt.show()

    img_predict = image.load_img(r'C:\Users\Of Corrupted Vision\Documents\Source Python\CNN mnist dataset\flow from directory version\Dataset Mod\Pygame Save files\digit.jpg', target_size = (28,28))
    img_predict = image.img_to_array(img_predict)
    img_predict = np.expand_dims(img_predict,axis=0)

    result = model.predict(img_predict)
    #print(result)

    if result[0][0] == 1:
        prediction = 'Number : 0'
    if result[0][1] == 1:
        prediction = 'Number : 1'
    if result[0][2] == 1:
        prediction = 'Number : 2'
    if result[0][3] == 1:
        prediction = 'Number : 3'
    if result[0][4] == 1:
        prediction = 'Number : 4'
    if result[0][5] == 1:
        prediction = 'Number : 5'
    if result[0][6] == 1:
        prediction = 'Number : 6'
    if result[0][7] == 1:
        prediction = 'Number : 7'
    if result[0][8] == 1:
        prediction = 'Number : 8'
    if result[0][9] == 1:
        prediction = 'Number : 9'

    print(prediction)
