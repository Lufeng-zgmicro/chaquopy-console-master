import os,sys

import numpy as np
# tensorflow
import tensorflow as tf

# audio
import librosa
import librosa.display

from tensorflow.keras.models import load_model

import matplotlib.pyplot as plt

from dataloader import extract_feature_from_framedata
from dataloader import FRAME_LEN, H_FRAME_NUM, W_FEATURE_DIM, FRAME_SHIFT_RATE
from tensorflow.keras.layers import Softmax

# use functions
# FRAME_LEN =  8000
# H_FRAME_NUM = 61
# W_FEATURE_DIM = 20


# predict long wav : functions

def sound_classifier_one_frame(frame_data, sr, model):

    assert sr == 16000, print("the sr == 16000!, pay attention !")

    mfcc = extract_feature_from_framedata(frame_data)
    train_x = mfcc[None]
    train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], train_x.shape[2], 1)

    predict = model.predict(train_x)

    predict_labels = np.argmax(predict, axis=1)

    # ret_out = predict[predict_labels.item()]

    return predict_labels[0]


def wav_prediction(filename, model):

    frame_shift = 0.5
    x, sr = librosa.load(filename, sr=16000)
    all_results = []
    pop_sound_times = []
    frame_num = np.floor((len(x) - (1-frame_shift) * FRAME_LEN) / (frame_shift*FRAME_LEN))
    frame_num = int(frame_num)
    for i in range(frame_num):
        st_idx = int(i * (frame_shift*FRAME_LEN))
        ed_idx = int(st_idx + FRAME_LEN)
        frame = x[st_idx:ed_idx]
        predict_result = sound_classifier_one_frame(frame, sr, model)
        if predict_result !=0 :
            pop_sound_times.append(st_idx / sr)
        all_results.append(predict_result)
    return all_results, pop_sound_times


def display_for_me(wav_path, predict_results):

    x, sr = librosa.load(wav_path, sr=16000)
    frame_shift = 0.5
    frame_num = np.floor((len(x) - (1 - frame_shift) * FRAME_LEN) / (frame_shift * FRAME_LEN))
    frame_num = int(frame_num)

    # y = []
    # tmp = np.ones((FRAME_LEN,))
    # last_frame = np.zeros((FRAME_LEN,))
    # cur_frame = last_frame
    # mid_idx = int(frame_shift * FRAME_LEN)
    # for i in range(frame_num):
    #     cur_frame[0:mid_idx] = last_frame[mid_idx:FRAME_LEN] + tmp[0:mid_idx] * predict_results[i]
    #     y.extend(cur_frame[0:mid_idx])
    #     last_frame = cur_frame

    y = []
    mid_idx = int(frame_shift * FRAME_LEN)
    tmp = np.ones((mid_idx,))
    for i in range(frame_num):
        cur_frame = tmp * predict_results[i]
        y.extend(cur_frame)

    x_display_len = frame_num * mid_idx
    plt.figure('pop predict result')
    plt.plot(x[0:x_display_len], color='b', label='wav')
    plt.plot(y, color='r', label='predict')
    plt.show()


import getopt

def parse_argv(argv):

    inputfile = ''
    outputfile = ''
    try:
        opts, args = getopt.getopt(argv, "hi:o:", ["ifile=", "ofile="])
    except getopt.GetoptError:
        print('test.py -i <inputfile> -o <outputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('pop_sound_check.py -i <wav_folder> -o <out_file_folder>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
    print('输入的目录为：', inputfile)
    print('输出的目录为：', outputfile)
    return inputfile, outputfile


if __name__ == "__main__":

    # in_wav_path, result_path = parse_argv(sys.argv[1:])
    # # qa give wav path
    in_wav_path = './demo_check-1'
    result_path = in_wav_path

    # proc
    sounds = [wav for wav in os.listdir(in_wav_path) if wav.endswith('.wav')]

    if len(sounds) != 0:

        # load model
        model = load_model('models/audio_tone.h5')

        f_pop = open(os.path.join(result_path, 'pop_sound_check_results.txt'), "wb")

        for idx,sound in enumerate(sounds):

            # if sound.endswith('OUT_pb-20210417-140908F6_Single_083.wav'):
            # if sound.endswith('tone.wav'):
            
            if True:
                results, pop_times = wav_prediction(os.path.join(in_wav_path, sound), model)
                display_for_me(os.path.join(in_wav_path, sound), results)

                # write file
                str_line = '{}-- '.format(idx)+os.path.join(in_wav_path, sound) + ':have ==== {} pop sounds\n'.format(len(pop_times))
                f_pop.write(str_line.encode())
                for i,t in enumerate(pop_times):
                    hour = t // 3600
                    min = (t - hour * 3600) // 60
                    sec = (t - hour * 3600 - min * 60)
                    f_pop.write('       |-- time{0}: {1} hour-- {2} min-- {3} second\n'.format(i, hour, min, sec).encode())

                # break
        f_pop.close()
    else:
        print(in_wav_path + ": don't have '.wav' file")

    print('Proc OVER!')

