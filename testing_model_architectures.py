# File to keep all the models I have tried



import keras
import tensorflow as tf



def rounded_accuracy(y_true, y_pred):
    return keras.metrics.binary_accuracy(tf.round(y_true), tf.round(y_pred))


############################################################
# Model 1
############################################################
resample_x, resample_y = 144
resample_z = 3
deep_e = keras.models.Sequential()
deep_e.add(keras.layers.Reshape([resample_x, resample_y, resample_z], input_shape=(resample_x, resample_y, resample_z)))
deep_e.add(keras.layers.Conv2D(16, kernel_size=3, padding='same', activation='selu'))
deep_e.add(keras.layers.MaxPool2D(pool_size=2))
deep_e.add(keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='selu'))
deep_e.add(keras.layers.MaxPool2D(pool_size=2))
deep_e.add(keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='selu'))
deep_e.add(keras.layers.MaxPool2D(pool_size=2))
deep_e.add(keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='selu'))
deep_e.add(keras.layers.MaxPool2D(pool_size=2))
# maybe add a flatten layer here
deep_e.add(keras.layers.Dense(30))
# 9,9, 30
deep_d = keras.models.Sequential()
deep_d.add(keras.layers.Dense(30))
deep_d.add(keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='selu'))
deep_d.add(keras.layers.UpSampling2D((2,2)))
deep_d.add(keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='selu'))
deep_d.add(keras.layers.UpSampling2D((2,2)))
deep_d.add(keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='selu'))
deep_d.add(keras.layers.UpSampling2D((2,2)))
deep_d.add(keras.layers.Conv2D(16, kernel_size=3, padding='same', activation='selu'))
deep_d.add(keras.layers.UpSampling2D((2,2)))
deep_d.add(keras.layers.Conv2D(3, (3,3), activation='sigmoid', padding='same'))
#deep_d.add(keras.layers.Reshape([resample_x, resample_y, resample_z]))
# Output dim is also 100, 100, 3 so: 30000
# Putting it all together
deep_ae = keras.models.Sequential([deep_e, deep_d])
# need pixel by pixel loss function here most likely
deep_ae.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(), metrics=[rounded_accuracy])
# Accuracy 89%


############################################################
# Model 2
############################################################
deep_e = keras.models.Sequential()
deep_e.add(keras.layers.Reshape([resample_x, resample_y, resample_z], input_shape=(resample_x, resample_y, resample_z)))
deep_e.add(keras.layers.Conv2D(16, kernel_size=3, padding='same', activation='selu'))
deep_e.add(keras.layers.MaxPool2D(pool_size=2))
deep_e.add(keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='selu'))
deep_e.add(keras.layers.MaxPool2D(pool_size=2))
deep_e.add(keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='selu'))
deep_e.add(keras.layers.MaxPool2D(pool_size=2))
deep_e.add(keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='selu'))
deep_e.add(keras.layers.MaxPool2D(pool_size=2))
# maybe add a flatten layer here
deep_e.add(keras.layers.Conv2D(4, kernel_size=3, padding='same', activation='selu'))
# deep_e.add(keras.layers.Dense(30))
# 9,9, 4
deep_d = keras.models.Sequential()
deep_d.add(keras.layers.Conv2D(4, kernel_size=3, padding='same', activation='selu'))
deep_d.add(keras.layers.UpSampling2D())
deep_d.add(keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='selu'))
deep_d.add(keras.layers.UpSampling2D((2,2)))
deep_d.add(keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='selu'))
deep_d.add(keras.layers.UpSampling2D((2,2)))
deep_d.add(keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='selu'))
deep_d.add(keras.layers.UpSampling2D((2,2)))
deep_d.add(keras.layers.Conv2D(16, kernel_size=3, padding='same', activation='selu'))
deep_d.add(keras.layers.Conv2D(3, (3,3), activation='sigmoid', padding='same'))
#deep_d.add(keras.layers.Reshape([resample_x, resample_y, resample_z]))
# Accuracy 89.5%



# Output dim is also 100, 100, 3 so: 30000

# Putting it all together
deep_ae = keras.models.Sequential([deep_e, deep_d])
# need pixel by pixel loss function here most likely
deep_ae.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(), metrics=[rounded_accuracy])

deep_ae.build(input_shape=(resample_x, resample_y, resample_z))


############################################################
# Model 3
############################################################

deep_e = keras.models.Sequential()
deep_e.add(keras.layers.Reshape([resample_x, resample_y, resample_z], input_shape=(resample_x, resample_y, resample_z)))
deep_e.add(keras.layers.Conv2D(16, kernel_size=3, padding='same', activation='selu'))
deep_e.add(keras.layers.MaxPool2D(pool_size=2))
deep_e.add(keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='selu'))
deep_e.add(keras.layers.MaxPool2D(pool_size=2))
deep_e.add(keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='selu'))
deep_e.add(keras.layers.MaxPool2D(pool_size=2))
deep_e.add(keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='selu'))
deep_e.add(keras.layers.MaxPool2D(pool_size=2))
# maybe add a flatten layer here
deep_e.add(keras.layers.Conv2D(4, kernel_size=3, padding='same', activation='selu'))
deep_e.add(keras.layers.MaxPool2D(pool_size=3))
deep_e.add(keras.layers.Dense(36))
# 9,9, 30
deep_d = keras.models.Sequential()
deep_d.add(keras.layers.UpSampling2D((3,3)))
deep_d.add(keras.layers.Conv2D(4, kernel_size=3, padding='same', activation='selu'))
deep_d.add(keras.layers.UpSampling2D())
deep_d.add(keras.layers.Conv2D(128, kernel_size=3, padding='same', activation='selu'))
deep_d.add(keras.layers.UpSampling2D((2,2)))
deep_d.add(keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='selu'))
deep_d.add(keras.layers.UpSampling2D((2,2)))
deep_d.add(keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='selu'))
deep_d.add(keras.layers.UpSampling2D((2,2)))
deep_d.add(keras.layers.Conv2D(16, kernel_size=3, padding='same', activation='selu'))
deep_d.add(keras.layers.Conv2D(3, (3,3), activation='sigmoid', padding='same'))
#deep_d.add(keras.layers.Reshape([resample_x, resample_y, resample_z]))




# Output dim is also 100, 100, 3 so: 30000

# Putting it all together
deep_ae = keras.models.Sequential([deep_e, deep_d])
# need pixel by pixel loss function here most likely
deep_ae.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(), metrics=[rounded_accuracy])

deep_ae.build(input_shape=(resample_x, resample_y, resample_z))

deep_ae.summary()

# 85% rounded_accuracy


cnn_epochs = 10
cnn_batch_size = 64

cnn_step_per_epoch = len_x_train / cnn_batch_size
print(cnn_step_per_epoch)
step_per_epoch = cnn_step_per_epoch
# hi

# model 4
cnn_model = Sequential()
cnn_model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
cnn_model.add(Conv2D(64, (3, 3), activation='relu'))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
#cnn_model.add(Dropout(0.25))
cnn_model.add(Flatten())
cnn_model.add(Dense(128, activation='relu'))
#cnn_model.add(Dense(128, activation='relu'))
#cnn_model.add(Dropout(0.5))
cnn_model.add(Dense(num_classes, activation='softmax'))
opt = keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9,momentum=0.0, epsilon=1e-07, centered=False, name="RMSprop")

cnn_model.compile(loss="sparse_categorical_crossentropy",
              optimizer=opt,
              metrics=['accuracy'])
history_cnn = cnn_model.fit(datagen.flow(x_train,training_labels,shuffle=True),
                                  epochs=cnn_epochs, steps_per_epoch=steps_per_epoch)
