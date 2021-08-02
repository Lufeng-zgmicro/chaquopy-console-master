
import os
import glob
import pickle
import numpy as np
import random
import math
import sklearn
import shutil

from scipy import signal
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt
import librosa
from keras.utils import to_categorical

FRAME_LEN = 8000        #sr = 16000, 0.5s: 0.5x16000=8000
H_FRAME_NUM =  167   	#61        #101
W_FEATURE_DIM = 128  	#20      #128
FRAME_SHIFT_RATE = 0.3


def get_file_list(wav_folder, class_id):

	file_list = []
	for wav_name in os.listdir(wav_folder):
		if wav_name.endswith('.wav'):
			wav_path = os.path.join(wav_folder, wav_name)
			file_list.append((wav_path, class_id))

	return file_list

def arrange_dataset(audio_path, non_audio_path):

	train_audio_path = audio_path
	train_none_audio_path = non_audio_path
	audio_list = get_file_list(train_audio_path, 1)
	none_audio_list = get_file_list(train_none_audio_path, 0)
	train_file_list = []
	train_file_list.extend(audio_list)
	train_file_list.extend(none_audio_list)

	return train_file_list


def my_spec_librosa(x, sr, shift_rate=0.5):

	frm_len = math.floor(10 * sr / 1000)
	frm_shift = math.floor(frm_len * (shift_rate))
	Nfft = int(np.power(2, np.ceil(np.log2(frm_len))))
	yy = librosa.stft(x, n_fft=Nfft, win_length=frm_len, hop_length=frm_shift, window='hamm')
	yy_magn = np.abs(yy)
	yy_magn = np.clip(yy_magn, 1e-20, 1e100)
	yy_magn = 20.0 * np.log10(yy_magn)

	if False:
		arr = np.flipud(yy_magn) # arr = arr[1:, 10:48]
	else:
		arr = yy_magn.T
		arr = arr[:, 1:]

	if False:
		arr = yy_magn[1:, 10:(yy_magn.shape[1]-10)]

	max_value = np.max(arr)
	arr = arr / max_value

	return arr

def display_spec(XFFT_all):
	print(XFFT_all.shape)
	plt.figure('wav spectrum')

	# arr = np.fliplr(XFFT_all)
	# plt.imshow(arr.T)
	plt.imshow(XFFT_all)
	plt.show()

def load_clip(filename, in_data_len):
	x, sr = librosa.load(filename, sr=16000)
	if x.shape[0] > in_data_len:
		x = x[0:in_data_len]
	else:
		x = np.pad(x,(0,in_data_len-x.shape[0]),'constant')
	return x, sr

def extract_feature(filename):

	if True:
	# if False:

		x, sr = load_clip(filename, FRAME_LEN)
		assert sr == 16000, print("the sr == 16000, pay attention! ")
		mfcc = my_spec_librosa(x, sr, FRAME_SHIFT_RATE)

	else:
		eps = 1e-15
		audio_wav, fs = librosa.load(filename, sr=16000)
		wav_len = FRAME_LEN

		if audio_wav.shape[0] < wav_len:
			empty_wav = np.zeros([wav_len])
			empty_wav[:audio_wav.shape[0]] = audio_wav
			audio_wav = empty_wav
		else:
			audio_wav = audio_wav[0:wav_len]

		spec_wav = librosa.stft(audio_wav, n_fft=256, hop_length=128, win_length=256, center=False)
		power_spec = np.abs(spec_wav) ** 2
		mel_spec = librosa.feature.melspectrogram(S=power_spec, sr=16000, n_fft=256, hop_length=128, power=2.0, n_mels=40,
												  fmin=20, fmax=7000)
		mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel_spec))

		# LBB_Keep_Tensorflow_Version
		mfcc = mfcc[0:20, :]
		mfcc = mfcc.T

	return mfcc

def extract_feature_from_framedata(audio_wav):

	if True:
	# if False:
		# x, sr = load_clip(filename, FRAME_LEN)
		mfcc = my_spec_librosa(audio_wav, 16000, FRAME_SHIFT_RATE)

	else:

		# eps = 1e-15
		# audio_wav, fs = librosa.load(filename, sr=16000)

		wav_len = FRAME_LEN

		if audio_wav.shape[0] < wav_len:
			empty_wav = np.zeros([wav_len])
			empty_wav[:audio_wav.shape[0]] = audio_wav
			audio_wav = empty_wav
		else:
			audio_wav = audio_wav[0:wav_len]

		spec_wav = librosa.stft(audio_wav, n_fft=256, hop_length=128, win_length=256, center=False)
		power_spec = np.abs(spec_wav) ** 2
		mel_spec = librosa.feature.melspectrogram(S=power_spec, sr=16000, n_fft=256, hop_length=128, power=2.0, n_mels=40,
												  fmin=20, fmax=7000)
		mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel_spec))

		# LBB_Keep_Tensorflow_Version
		mfcc = mfcc[0:20, :]
		mfcc = mfcc.T

	return mfcc

def generate_dataset(filenames):

	row_num = H_FRAME_NUM     #features_params['row_end_pos'] - features_params['row_start_pos']
	col_num = W_FEATURE_DIM   # features_params['col_end_pos'] - features_params['col_start_pos']

	features, labels = np.empty((0, row_num, col_num)), np.empty(0)
	cnt = 0
	cnt_all = len(filenames)

	for wavpath_label in filenames:
		mfccs = extract_feature(wavpath_label[0])
		features = np.append(features, mfccs[None], axis=0)
		cnt += 1
		if (cnt % 100 == 0):
			print([str(cnt) + ' / ' + str(cnt_all) + ' finished'])

		labels = np.append(labels, wavpath_label[1])
		# print(labels)

	return np.array(features), np.array(labels, dtype=np.int)


def generate_validate_batch_from_path(file_list, val_num):
	data_num = len(file_list)
	# indices = list(range(data_num))
	# random.shuffle(indices)
	try:
		if val_num <= data_num:
			batch_list = file_list[0:val_num]
			train_x, label_x = generate_dataset(batch_list)
			train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], train_x.shape[2], 1)
			label_x = to_categorical(label_x, num_classes=2)
			return train_x, label_x

	except IOError:
		print(" data_num <  batch_size! ")

	else:
		print("generate the  'batch data' OK")

	return



def generate_batch_from_path(dataset_path_list, batch_size, class_num):

	shuffle_flg = True

	data_num = len(dataset_path_list)
	indices = list(range(data_num))

	if shuffle_flg:
		random.shuffle(indices)
	try:
		batch_num = data_num // batch_size
		# for i in range(batch_num):
		cnt = 0
		while 1:
			# idx = indices[i * batch_size:i * batch_size + batch_size]
			if cnt >= batch_num:
				cnt = 0
			st = cnt * batch_size
			ed = st + batch_size
			cnt = cnt + 1
			print("     count:" + str(cnt))

			idx = indices[st:ed]
			# batch_path_list = dataset_path_list[idx]
			batch_path_list = []
			for index in idx:
				batch_path_list.append(dataset_path_list[index])

			train_x, label_x = generate_dataset(batch_path_list)
			train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], train_x.shape[2], 1)
			label_x = to_categorical(label_x, num_classes=class_num)
			yield train_x, label_x

	except IOError:
		print(" data_num <  batch_size! ")

	else:
		print("generate the  'batch data' OK")


if __name__ == '__main__':

	print("the file == dataloader...")

