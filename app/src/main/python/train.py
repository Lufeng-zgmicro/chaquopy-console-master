
from dataloader import arrange_dataset, generate_batch_from_path
from model import Audio_Net, show_history



# if __name__ == '__main__':
def train_function():

	batch_size = 32

	# load train data
	train_audio_path = r'./data/train/tone'
	train_none_audio_path = r'./data/train/non_tone'
	train_file_list = arrange_dataset(train_audio_path, train_none_audio_path)

	# load val data
	val_audio_path = r'./data/val/tone'
	val_none_audio_path = r'./data/val/non_tone'
	val_file_list = arrange_dataset(val_audio_path, val_none_audio_path)

	model = Audio_Net()

	# train_func(model, num_epochs)
	print('model.fit ====================================')
	history = model.fit_generator(generate_batch_from_path(train_file_list, batch_size=batch_size,class_num=2),
		                          steps_per_epoch=len(train_file_list) // batch_size, epochs= 15)


	save_model_dir = './models/audio_tone.h5'
	model.save(save_model_dir)
	print("save model.h5 ok...")

	# show_history(history)

	print('proc over!')
























