from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras import *
from keras.models import *
from keras.layers import Input, Dense
from keras.preprocessing import image
from keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions
#from keras.applications.inception_v3 import preprocess_input, decode_predictions
import numpy as np
from keras.utils.np_utils import to_categorical
import glob
import os.path
from keras.models import load_model

categorie =["cats","dogs","horses","humans"]

img_path = 'human.jpg'



def load_data(path):
    x_train = []
    y_train = []

    images = glob.glob(path+"/**/*")
    for photo in images:
        img = image.load_img(photo, target_size=(299, 299))
        tr_x = image.img_to_array(img)
        tr_x = preprocess_input(tr_x)
        label = (photo.split("/"))[1]
        label_place = categorie.index(label)

        x_train.append(tr_x)
        y_train.append(label_place)
    
    return np.array(x_train), to_categorical(y_train)

X_train, Y_train = load_data("dataset")

print(type(Y_train))
print(Y_train.shape)    # 808,4
print(X_train.shape)    # 808,299,299,3

input = Input(shape=(299, 299, 3))
#print(X_train.shape)
#raise
if (os.path.isfile("my_model.h5")):
    print("Modello esistente")
    model = load_model("my_model.h5")
else:
    print("Modello non presente, procedo al training")
    base_model = InceptionResNetV2(include_top=False, weights='imagenet', input_tensor=input, input_shape=(299, 299, 3), pooling='avg', classes=1000)
    for l in base_model.layers:
        l.trainable = False

    t = base_model(input)
    o = Dense(len(categorie), activation='softmax')(t)
    model = Model(inputs=input, outputs=o)


    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    

    model.fit(X_train, Y_train,
                  batch_size=32,
                  epochs=3,
                  shuffle=True,
                  verbose=1
                  )

    model.save("my_model.h5")

print(model.summary())
img = image.load_img(img_path, target_size=(299, 299))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x,batch_size=None, verbose=0)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print("PREDICTIONS: (cat | dog | horse | human")
print(preds)
