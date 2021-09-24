# THIS ENTIRE CODE WAS EXECUTED USING GOOGLE COLABORATORY IN ORDER TO AVOID INCOMPATIBILITY ISSUES

# LOAD LIBRARIES
%tensorflow_version 2.x
!pip install -q -U keras-tuner

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import keras_tuner as kt

# INITIALIZE GPU
from tensorflow.python.client import device_lib

device = tf.test.gpu_device_name()

if (device != ''):
  print('GPU is found at {}'.format(device))
  print(device_lib.list_local_devices())
else:
  print('GPU not found')

# DEFINE FUNCTIONS
BATCH_SIZE = 64

def load_train_data(filepath):
  train_data = tf.keras.preprocessing.image_dataset_from_directory(
      filepath,
      label_mode='binary',
      validation_split=0.2,
      subset='training',
      seed=123,
      batch_size=BATCH_SIZE,
      image_size=(224, 224)
      )
  return train_data

def load_val_data(filepath):
  val_data = tf.keras.preprocessing.image_dataset_from_directory(
      filepath,
      label_mode='binary',
      validation_split=0.2,
      subset='validation',
      seed=123,
      batch_size=BATCH_SIZE,
      image_size=(224, 224)
      )
  return val_data

def create_mobnet(hp):
  hp_dropout_rate = hp.Float('dropout', 0.0, 0.9, step=0.1, default=0.5)
  model = tf.keras.applications.MobileNet(dropout=hp_dropout_rate)

  base_input = model.layers[0].input
  base_output = model.layers[-4].output

  flat_layer = layers.Flatten()(base_output)
  dense_layer = layers.Dense(1)(flat_layer)
  activation_layer = layers.Activation('sigmoid')(dense_layer)
  
  model = tf.keras.Model(inputs=base_input, outputs=activation_layer)

  hp_learning_rate=hp.Float('learning_rate', 0.0001, 0.001, step=0.0001, default=0.0005)

  model.compile(
      loss="binary_crossentropy",
      optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
      metrics=['accuracy']
      )
  return model

# LOAD DATA
train_data = load_train_data('/content/drive/MyDrive/p80_eye_dataset')
val_data = load_val_data('/content/drive/MyDrive/p80_eye_dataset')

# DATA VISUALIZATION
class_names = train_data.class_names
print(class_names)

['Close_Eyes', 'Open_Eyes']

for image_batch, labels_batch in train_data:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

plt.figure(figsize=(10, 10))
for images, labels in train_data.take(1):
  for i in range(25):
    ax = plt.subplot(5, 5, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[np.squeeze(labels[i]).astype("int32")])
    plt.axis("off")

# NORMALIZE TRAINING DATA
normalize_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
normalized_train_data = train_data.map(lambda x,y: (normalize_layer(x),y))
normalized_val_data = val_data.map(lambda x,y: (normalize_layer(x),y))

image_batch, labels_batch = next(iter(normalized_train_data))
first_image = image_batch[0]
first_label = labels_batch[0]
print(np.array(first_image))
print(np.array(first_label))

# HYPERPARAMETERS TUNING
tuner = kt.BayesianOptimization(create_mobnet,
                                objective='val_accuracy',
                                max_trials=10)

tuner.search_space_summary()

tuner.search(normalized_train_data,
             validation_data=normalized_val_data,
             epochs=30,
             callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, mode='auto', verbose=1, restore_best_weights=True)])

tuner.search(normalized_train_data,
             validation_data=normalized_val_data,
             epochs=30,
             callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, mode='auto', verbose=1, restore_best_weights=True)])

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
dropout = best_hps.get('dropout')
lrate = best_hps.get('learning_rate')
print(f"""Best Hyperparameters Value:
Dropout Rate: {dropout}
Learning Rate : {lrate}
""")

best_model = tuner.get_best_models(num_models=1)[0]
best_model.save_weights('/content/drive/MyDrive/fp-ddd/saved_model/mobnet_drowsy_v8-1_weights.h5')
best_model.save('/content/drive/MyDrive/fp-ddd/saved_model/mobnet_drowsy_v8-1.h5')
