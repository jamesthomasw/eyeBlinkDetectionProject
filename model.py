import tensorflow as tf
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

def create_model():
    
    model = tf.keras.applications.MobileNet(include_top=False,
                                            input_shape=(128, 128, 3),
                                            weights='imagenet',
                                            dropout=0.3)
    
    base_input = model.layers[0].input
    base_output = model.layers[-4].output

    flat_layer = layers.Flatten()(base_output)
    dense_layer = layers.Dense(1)(flat_layer)
    activation_layer = layers.Activation('sigmoid')(dense_layer)

    model = tf.keras.Model(inputs=base_input, outputs=activation_layer)

    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    
    return model

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
    except RuntimeError as e:
        print(e)

source_dir = '/Users/James Thomas/VScodeProjects/Eye Blink Detection/CEW_Eyes_Dataset/'
categories = ['ClosedEyes', 'OpenEyes']
img_size = 128
eye_data = []

for category in categories:
    path = os.path.join(source_dir, category)
    class_num = categories.index(category)
    for img in os.listdir(path): 
        try:
            image = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            gray = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            gray_resized = cv2.resize(gray, (img_size, img_size))
            eye_data.append([gray_resized, class_num])
        except Exception as e:
            print(e)

import random
random.shuffle(eye_data)

X = []
y = []

for image, label in eye_data:
    X.append(image)
    y.append(label)
    
X = np.array(X).reshape(-1, img_size, img_size, 3)
X = X/255.0
y = np.array(y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

model = create_model()

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

checkpoint = ModelCheckpoint(
    "saved_model/mobilenet_128_v2.h5",
    monitor='val_accuracy',
    verbose=1,
    save_best_only=True,
    save_weights_only=False,
    mode='auto'
)

early_stop = EarlyStopping(
    monitor='val_accuracy',
    restore_best_weights=True,
    min_delta=0,
    patience=5,
    verbose=1,
    mode='auto'
)

history = model.fit(
    x=X_train,
    y=y_train,
    validation_data=(X_test, y_test),
    batch_size=32,
    epochs=20,
    callbacks=[checkpoint, early_stop]
)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Training Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train_accuracy', 'val_accuracy'], loc='upper left')
plt.savefig('saved_figure/plot_128-2a.png', dpi=300, bbox_inches='tight')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Training Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train_loss', 'val_loss'], loc='lower left')
plt.savefig('saved_figure/plot_128-2b.png', dpi=300, bbox_inches='tight')
plt.show()
