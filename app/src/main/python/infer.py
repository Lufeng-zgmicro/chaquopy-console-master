
import numpy as np
from tensorflow.keras.models import load_model
from dataloader import arrange_dataset, generate_validate_batch_from_path, extract_feature
from model import Audio_Net, show_history


def get_accurate(model, val_x, val_y):

	correct = [0, 0]
	total = [0, 0]

	predicted = model.predict(val_x)
	predict_class = np.argmax(predicted, axis=1)
	true_class = np.argmax(val_y, axis=1)

	for i, lab in enumerate(predict_class):
		total[true_class[i]] += 1
		correct[true_class[i]] += (lab == true_class[i])

	val_err_rate = correct[0] / total[0]
	val_acc_rate = correct[1] / total[1]

	acc_rate = (correct[0] + correct[1]) / (total[0] + total[1])

	print("the all acc rate == > {}%".format(acc_rate*100))
	print('val_acc_num: {0}, calcu accurate: {1} % '.format(total[1], 100 * val_acc_rate))
	print('val_err_num: {0}, calcu accurate: {1} % '.format(total[0], 100 * val_err_rate))

	return

def infer_one_wav(wav_path, model):

	mfccs = extract_feature(wav_path)
	train_x = mfccs[None]
	train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], train_x.shape[2], 1)

	predicted = model.predict(train_x)

	predict_ret = np.argmax(predicted, axis=1)

	print("the wav check out ==> ", predict_ret)

	return


if __name__ == '__main__':

	if True:
	# if False:

		# load val data
		val_audio_path = r'./data/val/tone'
		val_none_audio_path = r'./data/val/non_tone'
		val_file_list = arrange_dataset(val_audio_path, val_none_audio_path)

		model = load_model('models/audio_tone.h5')
		# model = load_model('demo_models/audio_tone--good.h5')

		# generate npy
		val_x, val_y = generate_validate_batch_from_path(val_file_list, len(val_file_list))

		get_accurate(model, val_x, val_y)

	else:

		wav_path = './data/check_tone.wav'
		model = load_model('models/audio_tone.h5')
		infer_one_wav(wav_path, model)

	print("infer over...")



