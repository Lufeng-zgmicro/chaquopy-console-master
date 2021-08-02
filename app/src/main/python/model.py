
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from dataloader import FRAME_LEN, H_FRAME_NUM, W_FEATURE_DIM

# FRAME_LEN = 8000    #sr = 16000, 10ms
# H_FRAME_NUM = 61    #101
# W_FEATURE_DIM = 20  #128
# FRAME_SHIFT_RATE = 0.5


def Audio_Net(num_classes=2):

	inputs = layers.Input(shape=(H_FRAME_NUM, W_FEATURE_DIM, 1))

	#conv2d
	x = layers.Convolution2D(16, kernel_size=(5, 10), strides=(2, 2), activation='relu')(inputs)
	x = layers.MaxPooling2D((1, 2))(x)

	# x = layers.Convolution2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu')(x)
	# x = layers.MaxPooling2D((1, 2))(x)

	if True:
	# if False:
		# gru
		x = layers.Reshape((82, 480))(x)
		x = layers.GRU(32, return_sequences=False, activation='tanh', dropout=0.3)(x)
		x = layers.Dropout(0.5)(x)

		# fc
		# x = layers.Dense(32)(x)
		# x = layers.ReLU()(x)
		# x = layers.Dropout(0.3)(x)

		# x = layers.Dense(6)(x)
		# x = layers.ReLU()(x)
	else:
		x = layers.Flatten()(x)
		# x = layers.Dense(64, activation='relu')(x)

	x = layers.Dense(num_classes)(x)
	preds = layers.Softmax()(x)

	model = tf.keras.Model(inputs=inputs, outputs=preds)

	model.compile(optimizer=tf.keras.optimizers.Adam(0.0001),
				  loss=tf.keras.losses.categorical_crossentropy,
				  metrics=['accuracy'])

	model.summary(line_length=80)   # (80 +1) * 16 * 3 + (16 + 1) * 16 * 3

	return model



	# model = Sequential()
	#
	# model.add(Convolution2D(16, (5, 5), activation='relu', input_shape=(row_num, col_num, 1)))
	# model.add(MaxPooling2D((2, 2)))
	# model.add(Dropout(0.5))
	#
	# model.add(Convolution2D(16, (3, 3), activation='relu'))
	# model.add(MaxPooling2D((2, 2)))
	# model.add(Dropout(0.5))
	#
	# model.add(TimeDistributed(Flatten()))
	# model.add(GRU(64))
	#
	# model.add(Dense(64, activation='relu'))
	# model.add(Dense(class_num, activation='softmax'))
	#
	#
	# # loss & opt
	# model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
	# # model.optimizer.lr.assign(0.000001)
	# # print model
	# model.summary(line_length=80)


def show_history(history):
	print(history.history.keys())
	fig = plt.figure(figsize=(20, 5))
	plt.subplot(121)
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.subplot(122)
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='lower left')
	plt.show()

	return


