# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
#import pydub
import numpy as np
import math
from sklearn.metrics.pairwise import cosine_similarity
from os import listdir
from os.path import isfile, join
from scipy.spatial.distance import cdist

import librosa



reduced_array_size = 2000

"""MP3 to numpy array"""

# mp3_file = open("thunderstorm.mp3", "rb")
# a = pydub.AudioSegment.from_mp3(mp3_file)
# y = np.array(a.get_array_of_samples())
# print(y)
def cos_sim_2d(x, y):
    norm_x = x / np.linalg.norm(x, axis=1, keepdims=True)
    norm_y = y / np.linalg.norm(y, axis=1, keepdims=True)
    return np.matmul(norm_x, norm_y.T)

def read(f, normalized=False):
    """MP3 to numpy array"""
    a = pydub.AudioSegment.from_mp3(f)
    y = np.array(a.get_array_of_samples())
    if a.channels == 2:
        y = y.reshape((-1, 2))
    if normalized:
        return a.frame_rate, np.float32(y) / 2**15
    else:
        return a.frame_rate, y

def analyze_song(songfile):
    x, sr = librosa.load('songs/' + songfile,
                         sr=None)  # loads an audio file and decodes it into a 1-dimensional array which is a time series x , and sr is a sampling rate of x

    freqs = np.fft.fftfreq(x.size)
    freqs_median = np.median(freqs) # or better yet would be the mean
    # print(freqs)
    # print(freqs_median)
    # print("Analyzing the song: " + songfile)

    zeros = librosa.zero_crossings(x)
    zeros_sum = np.sum(zeros)
    # print("zeros sum")
    # print(sum(zeros))
    # print("feature")
    feature = np.mean(librosa.feature.poly_features(y=x))
    # print(feature)
    # print("root square mean energy")
    rmse = librosa.feature.rms(y=x)[0]
    rmse_mean = np.mean(rmse)
    # print(np.mean(rmse))
    # print("fundamental frequency")
    f0, voiced_flag, voiced_probs = librosa.pyin(x,
                                                 fmin=librosa.note_to_hz('C2'),
                                                 fmax=librosa.note_to_hz('C7'))
    times = librosa.times_like(f0)
    # print(f0)
    # print("fundamental frequency times")
    # print(times)
    # print("fundamental frequency times length")
    # print(len(times))

    f0 = librosa.yin(x, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    mean_fundamental_frequency = np.mean(f0)
    # print("Fundamental frequency yin shape")
    # print(f0.shape)
    # print("Fundamental frequency yin")
    # print(f0)
    # print("Fundamental frequency yin mean")
    # print(mean_fundamental_frequency)

    mean_tempo = np.mean(librosa.beat.tempo(y=x)[0])
    # print("tempo")
    # print(tempo)

    hop_length = 512
    y = x
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr, hop_length=hop_length)
    mean_tempogram = np.mean(tempogram)
    # print("tempogram mean")
    # print(mean_tempogram)


    return zeros_sum, rmse_mean, mean_tempo, mean_fundamental_frequency, mean_tempogram #returns a tuple with numerized statistically reduced song features


def compare_two_songs(song1_data, song2_data):

    # print("Enter the first song to compare (full name with dot and extension):")
    # song1 = input()
    # print("Enter the second song to compare (full name with dot and extension):")
    # song2 = input()

    #sr, x = read('songs/' + songfile1)
    # cosine  = cosine_similarity([song1_data], [song1_data])
    similarity = cdist([song1_data], [song2_data], 'euclidean')
    #euclidean = np.linalg.norm(song1_slice - song2_slice)

    # if (similarity[0][0] > 0):
    #     print("The songs: " + str(songfile1) + " " + str(songfile2) + " " + " are similar")
    # else:
    #     print("The songs: " + str(songfile1) + " " + str(songfile2) + " " + " are not similar")
    return np.average(similarity)



def analyze_all_songs():

    print("Enter the name of the audio file to compare: ")
    song1 = input()
    song1_features = np.array(analyze_song(song1), dtype=object)
    print("array1")
    print(song1_features)
    print("array1 end")

    for songfile in listdir("songs"):
        if isfile(join("songs", songfile)):
            song2_features =  np.array(analyze_song(songfile))
            similarity = compare_two_songs(song1_features, song2_features)
            print("Similarity " + song1 + " " + songfile + " " + str(similarity))






    # for songfile in listdir("songs"):
    #     if isfile(join("songs", songfile)):
    #         analyze_song(songfile)

# import audio2numpy as a2n
# x,sr=a2n.audio_from_file("thunderstorm.mp3")
# print(x[222])





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
       analyze_all_songs()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/


