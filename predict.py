# Load the image and resize it to the input size of the model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow import keras
import numpy as np
model=load_model('models')

image1 = load_img('TEST/cat_test.jpg', target_size=(150, 150), grayscale=True)
image2 = load_img('TEST/American_Eskimo_Dog.jpg',target_size=(150, 150), grayscale=True)
image3 = load_img('TEST/1200px-American-Eskimo-dog.jpg',
                  target_size=(150, 150), grayscale=True)

image_array = [image1,image2,image3]

for image in image_array:
    image=img_to_array(image)
    image=image / 255.0
    image=np.expand_dims(image, axis=0)
    prediction=model.predict(image)

    if prediction > 0.5:
        print("It's a dog")
    else:
        print("It's a cat")