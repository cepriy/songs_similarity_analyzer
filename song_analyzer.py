# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
#import pydub
import sys

import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
from scipy.spatial.distance import cdist
from os.path import exists
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
    x, sr = librosa.load(songfile,
                         sr=None)  # loads an audio file and decodes it into a 1-dimensional array which is a time series x , and sr is a sampling rate of x

    nonzero = np.count_nonzero(x)/10000000 # proportional to the duration of non-silent fragments, reduced by 10000000 times

    freqs = np.fft.fftfreq(x.size)

    freqs_median = np.median(freqs) # or better yet would be the mean
    # print(freqs)
    # print(freqs_median)
    # print("Analyzing the song: " + songfile)

    zeros = librosa.zero_crossings(x)
    zeros_sum = np.sum(zeros)/1000000 #division for normalizing
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
    mean_fundamental_frequency = np.mean(f0) / 1000
    # print("Fundamental frequency yin shape")
    # print(f0.shape)
    # print("Fundamental frequency yin")
    # print(f0)
    # print("Fundamental frequency yin mean")
    # print(mean_fundamental_frequency)

    mean_tempo = np.mean(librosa.beat.tempo(y=x)[0])/1000
    # print("tempo")
    # print(tempo)

    hop_length = 512
    y = x
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr, hop_length=hop_length)
    mean_tempogram = np.mean(tempogram)
    # print("tempogram mean")
    # print(mean_tempogram)

    X = librosa.stft(x)
    Xdb = librosa.amplitude_to_db(abs(X))
    mean_Xdb = np.mean(Xdb)/100

    spectral_centroids = librosa.feature.spectral_centroid(y=x, sr=sr)[0]
    mean_spectral = np.mean(spectral_centroids)/10000


    return zeros_sum , rmse_mean, mean_tempo, mean_fundamental_frequency, mean_tempogram, nonzero,\
           mean_Xdb, mean_spectral #returns a tuple with numerized statistically reduced song features


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



def analyze_all_songs(song_samples_dir):



    #print("Enter the name of the audio file to compare: ")
    # song1 = input()
    song1 = sys.argv[1]

    distance_method = 'euclidean'
    if (sys.argv[3]):
        distance_method = sys.argv[3]

    df1 = df = pd.DataFrame({"a": [1, 2, 3, 4],
                             "b": [5, 6, 7, 8]})

    # Creating the Second Dataframe using dictionary
    df2 = pd.DataFrame({"a": [1, 2, 3],
                        "b": [5, 6, 7]})

    df1 = df1.append(df2)
    print(df1)

    df_songs_features = pd.DataFrame({'song_name': 'a songfile',
                               'zeros_sum': 1,
                              'rmse_mean': [1],
                              'mean_tempo': [1],
                              'mean_fund_freq': [1],
                              'mean_tempogram':[1],
                              'non_zero': [1],
                              'mean_xDB': [1],
                              'mean_spectral': [1]})

    df_tmp2 = pd.DataFrame({'zeros_sum': [2],
                              'rmse_mean': [2],
                              'mean_tempo': [2],
                              'mean_fund_freq': [2],
                              'mean_tempogram': [2],
                              'non_zero': [2],
                              'mean_xDB': [2],
                              'mean_spectral': [2]})

    df_songs_features.append(df_tmp2)

    print(df_songs_features)



    #song1_features = np.array(analyze_song(song_samples_dir + '/' + song1), dtype=object)

    #song1_features =  (song1_features - np.min(song1_features)) / (np.max(song1_features) - np.min(song1_features))

    #df_songs_features = pd.DataFrame(np.zeros((1, 8)))

    print(df_songs_features.shape)
    for songfile in listdir(song_samples_dir):
        if isfile(join(song_samples_dir, songfile)):
            song2_features =  np.array(analyze_song(song_samples_dir + '/' + songfile))
            print("song2_features.shape")
            print(song2_features.shape)
            print(song2_features[0])
           # print(song2_features.reshape(1,8))


            #df_songs_features.append(pd.DataFrame(song2_features.reshape(1, 8), columns=list(df)), ignore_index=True)

            temp_df = pd.DataFrame({'song_name' : songfile,
                                    'zeros_sum': [song2_features[0]],
                                      'rmse_mean': [song2_features[1]],
                                      'mean_tempo': [song2_features[2]],
                                      'mean_fund_freq': [song2_features[3]],
                                      'mean_tempogram': [song2_features[4]],
                                      'non_zero': [song2_features[5]],
                                      'mean_xDB': [song2_features[6]],
                                      'mean_spectral': [song2_features[7]]})
            df_songs_features = df_songs_features.append(temp_df, ignore_index=True)
            #songs.append(songfile)
            print("Currrent dataframe with features")
            print(df_songs_features)
            # similarity = compare_two_songs(song1_features, song2_features)
            # #rates.append(rates)
            # print("Similarity " + song1 + " " + songfile + " " + str(similarity))
    df_songs_features.to_csv(song_samples_dir + "_features.csv")

    print("Resulting dataframe")
    print(df_songs_features)

    # for songfile in listdir("songs"):
    #     if isfile(join("songs", songfile)):
    #         analyze_song(songfile)

def get_current_song_comparison_scores(sample_scores_df, current_song_features):
    song_comparison_scores = pd.DataFrame({'song_name':[], 'score':[]} )

    for index, row in sample_scores_df.iterrows():
        current_score = compare_two_songs(row[2:10], current_song_features)
        song_comparison_scores = song_comparison_scores.append({'song_name': row['song_name'],'score': current_score},  ignore_index=True)

    song_comparison_scores = song_comparison_scores.sort_values(['score'], ascending=[False])
    return song_comparison_scores

# import audio2numpy as a2n
# x,sr=a2n.audio_from_file("thunderstorm.mp3")
# print(x[222])

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
       print("Command line arguments")
       print(sys.argv)

       song_samples_dir = 'songs'
       if (sys.argv[2] is not None):
           song_samples_dir = sys.argv[2]
           if (not exists (song_samples_dir + '_features.csv')):
               analyze_all_songs(song_samples_dir)
               print("CREATING DATASET! IT MAY TAKE A WHILE")
               print("THE DATASET ALREADY IS CREATED!!!")
               current_song_features = analyze_song(sys.argv[1])
               sample_scores_df = pd.read_csv(song_samples_dir + '_features.csv')
               get_current_song_comparison_scores(sample_scores_df, current_song_features)

           else:
               print("THE DATASET ALREADY EXISTS!!!")
               current_song_features = analyze_song(sys.argv[1])
               sample_scores_df = pd.read_csv(song_samples_dir + '_features.csv')
               score_df = get_current_song_comparison_scores(sample_scores_df, current_song_features)
               print(score_df)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/


