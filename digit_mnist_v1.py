import pygame
import time
import tensorflow as tf
import numpy as np
import cv2 as cv

screen_width = 500
screen_height = 500
black = (0,0,0)
white = (255,255,255)
height = 20
width = 20

time_taken = time.asctime(time.localtime(time.time()))
time_taken = time_taken.replace(" ","_")
time_taken = time_taken.replace(":",".")
save_file = "screenshots/" + time_taken + ".png"

def mnist_predict(input):
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train/255.0, x_test/255.0
    print(x_train.shape)
    print(x_test.shape)
    x_train = np.expand_dims(x_train, axis=3)
    x_test = np.expand_dims(x_test, axis=3)
    print(x_train.shape)
    print(x_test.shape)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.summary()
    model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics = ["accuracy"])
    model.fit(x_train, y_train, epochs = 1)
    model.evaluate(x_test, y_test)
    print(x_test.shape)
    # classifications = model.predict(x_test)
    # print(classifications[0])
    # print(y_test[0])
    custom_classifications = model.predict(input)
    print(custom_classifications[0])
    current = 0
    for index, item in enumerate(custom_classifications[0]):
        if item > current:
            current = item
            print(current)
            return index



def main():
    pygame.init()
    win = pygame.display.set_mode((screen_height,screen_width))
    pygame.display.set_caption("Digit MNIST Prediction by Syukri Ghazali")
    clock = pygame.time.Clock()
    running = True
    isPressed = False

    while running:
        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False      
            if event.type == pygame.MOUSEBUTTONDOWN:
                isPressed = True
            if event.type == pygame.MOUSEBUTTONUP:
                isPressed = False
                pygame.image.save(win, save_file)
                print("The screenshot has been saved at " + time_taken)

                img = cv.imread(save_file)
                dim = (28,28)
                img = cv.resize(img, dim, interpolation = cv.INTER_AREA)
                grayImage = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                img = np.expand_dims(grayImage, axis=2)
                img = np.expand_dims(img, axis=0)
                print(img.shape)
                output = mnist_predict(img)
                print("The predicted image is " + str(output))


        X_pos, Y_pos = pygame.mouse.get_pos()
        if(isPressed):
            pygame.draw.rect(win, white, (X_pos, Y_pos, width, height))

        clock.tick(200)
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
