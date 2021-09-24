import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt


def create_mobnet(hp_dropout_rate, hp_learning_rate):

    model = tf.keras.applications.MobileNet(dropout=hp_dropout_rate)

    base_input = model.layers[0].input
    base_output = model.layers[-4].output

    flat_layer = layers.Flatten()(base_output)
    dense_layer = layers.Dense(1)(flat_layer)
    activation_layer = layers.Activation('sigmoid')(dense_layer)

    model = tf.keras.Model(inputs=base_input, outputs=activation_layer)

    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
        metrics=['accuracy']
    )
    return model


def load_train_data():
    train_data = tf.keras.preprocessing.image_dataset_from_directory(
        'p80_eye_dataset',
        label_mode='binary',
        validation_split=0.2,
        subset='training',
        seed=123,
        batch_size=64,
        image_size=(224, 224)
    )
    return train_data


def load_val_data():
    val_data = tf.keras.preprocessing.image_dataset_from_directory(
        'p80_eye_dataset',
        label_mode='binary',
        validation_split=0.2,
        subset='validation',
        seed=123,
        batch_size=64,
        image_size=(224, 224)
    )
    return val_data


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
    except RuntimeError as e:
        print(e)

train_data = load_train_data()
val_data = load_val_data()

normalize_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
normalized_train_data = train_data.map(lambda x, y: (normalize_layer(x), y))
normalized_val_data = val_data.map(lambda x, y: (normalize_layer(x), y))

dropout = 0.0
learning_rate = 0.0007

model = create_mobnet(hp_dropout_rate=dropout, hp_learning_rate=learning_rate)

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "saved_model/mobnet_drowsy_v8-10.h5",
    monitor='val_accuracy',
    verbose=1,
    save_best_only=True,
    save_weights_only=False,
    mode='auto'
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    restore_best_weights=True,
    min_delta=0,
    patience=5,
    verbose=1,
    mode='auto'
)

history = model.fit(
    normalized_train_data,
    validation_data=normalized_val_data,
    epochs=30,
    callbacks=[model_checkpoint, early_stop]
)

model.save_weights("saved_model/mobnet_drowsy_v8-10_weights.h5")

loss, acc = model.evaluate(normalized_val_data, verbose=2)
print(f"""
Hold-out cross validation
Loss: {loss}
Accuracy: {100*acc}%
""")

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Training Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train_accuracy', 'val_accuracy'], loc='upper left')
plt.savefig('saved_figure/plot_8-10a.png', dpi=300, bbox_inches='tight')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Training Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train_loss', 'val_loss'], loc='lower left')
plt.savefig('saved_figure/plot_8-10b.png', dpi=300, bbox_inches='tight')
plt.show()