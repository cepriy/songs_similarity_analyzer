import sys
import numpy as np
import time
import pandas as pd
from os import listdir
import librosa
from os.path import isfile, join
from scipy.spatial.distance import cdist
from os.path import exists

def analyze_song(songfile):
    start_song_analysis = time.time()
    x, sr = librosa.load(songfile,
                         sr=None)  # loads an audio file and decodes it into a 1-dimensional array which is a time series x , and sr is a sampling rate of x

    print("Mp3 to nmpy array took" + str(time.time() - start_song_analysis) + " seconds")
    time_stamp_2 = time.time()
    nonzero = np.count_nonzero(x)/10000000 # proportional to the duration of non-silent fragments, reduced by 10000000 times
                                        # Nonzeros allow measuring the percussion characteristics
    print("Nonzero calculating took " + str(time.time() - time_stamp_2) + " seconds")
    time_stamp_3 = time.time()

    zeros = librosa.zero_crossings(x) # Zeros measure the silence
    zeros_sum = np.sum(zeros)/1000000 #division for normalizing
    print("Zeros took" + str(time.time() - time_stamp_3) + " seconds")
    time_stamp_4 = time.time()
    rmse = librosa.feature.rms(y=x)[0] # rmse corresponds to the energy level of an audio
    rmse_mean = np.mean(rmse)
    print("Rmse took" + str(time.time() - time_stamp_4) + " seconds")
    time_stamp_5 = time.time()
    f0 = librosa.yin(x, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')) #possible bottle neck
    mean_fundamental_frequency = np.mean(f0) / 1000
    #times = librosa.times_like(f0)


    print("F0 took" + str(time.time() - time_stamp_5) + " seconds")
    time_stamp_6 = time.time()

    print("Mean fundamental" + str(time.time() - time_stamp_6) + " seconds")
    time_stamp_7 = time.time()

    mean_tempo = np.mean(librosa.beat.tempo(y=x)[0])/1000
    print("Mean tempo took" + str(time.time() - time_stamp_7) + " seconds")
    time_stamp_8 = time.time()
    hop_length = 512
    y = x
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length) #Spectral flux is a measure of how quickly the power spectrum of a signal is changing
    print("Oenv took" + str(time.time() - time_stamp_8) + " seconds")
    time_stamp_9 = time.time()
    tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr, hop_length=hop_length)
    print("Tempogram took" + str(time.time() - time_stamp_9) + " seconds")

    mean_tempogram = np.mean(tempogram)
    time_stamp_10 = time.time()
    X = librosa.stft(x)
    Xdb = librosa.amplitude_to_db(abs(X))
    mean_Xdb = np.mean(Xdb)/100
    print("Xdb" + str(time.time() - time_stamp_10) + " seconds")
    time_stamp_11 = time.time()

    spectral_centroids = librosa.feature.spectral_centroid(y=x, sr=sr)[0]
    mean_spectral = np.mean(spectral_centroids)/10000
    print("Mean spectral took" + str(time.time() - time_stamp_11) + " seconds")


    return zeros_sum , rmse_mean, mean_tempo, mean_fundamental_frequency, mean_tempogram, nonzero,\
           mean_Xdb, mean_spectral #returns a tuple with numerized statistically reduced song features


def compare_two_songs(song1_data, song2_data, method):
    similarity = cdist([song1_data], [song2_data], method)
    return np.average(similarity)

def analyze_all_songs(song_samples_dir):
    df_songs_features = pd.DataFrame()

    print(df_songs_features.shape)
    for songfile in listdir(song_samples_dir):
        if isfile(join(song_samples_dir, songfile)):
            song2_features =  np.array(analyze_song(song_samples_dir + '/' + songfile))
            temp_df = pd.DataFrame({'song_name' : songfile,
                                    'zeros_sum': [song2_features[0]],
                                      'rmse_mean': [song2_features[1]],
                                      'mean_tempo': [song2_features[2]],
                                      'mean_fund_freq': [song2_features[3]],
                                      'mean_tempogram': [song2_features[4]],
                                      'non_zero': [song2_features[5]],
                                      'mean_xDB': [song2_features[6]],
                                      'mean_spectral': [song2_features[7]]})
            df_songs_features = pd.concat([df_songs_features, temp_df], ignore_index=True)

    df_songs_features.to_csv(song_samples_dir + "_features.csv")

    print("Resulting dataframe")
    print(df_songs_features)


def get_current_song_comparison_scores(sample_scores_df, current_song_features):
    song_comparison_scores = pd.DataFrame({'song_name':[], 'score':[]} )

    for index, row in sample_scores_df.iterrows():
        current_score = compare_two_songs(row[2:10].values.tolist(), current_song_features, sys.argv[3])
        print("current_score")
        print(current_score)
        df_with_cuurent_score = pd.DataFrame({'song_name': row['song_name'], 'score': [current_score]})

        song_comparison_scores = pd.concat([song_comparison_scores, df_with_cuurent_score], ignore_index=True)

    song_comparison_scores = song_comparison_scores.sort_values(['score'], ascending=[False])
    return song_comparison_scores

if __name__ == '__main__':
       print("Command line arguments")
       print(sys.argv)
       usage = """Incorrect command arguments! Usage:
              python3 song_analyzer.py <songfile> <song samples directory(optional)> <distance computing method(optional)>"""
       if (len(sys.argv) < 2 or len(sys.argv) > 4):
           print(usage)
       else:

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
                   start_time = time.time()

                   print("USING THE DEFAULT DATASET!!!")
                   current_song_features = analyze_song(sys.argv[1])
                   print("Current song analysis took " + str(time.time() - start_time) + " seconds")
                   print(current_song_features)
                   time_1 = time.time()
                   sample_scores_df = pd.read_csv(song_samples_dir + '_features.csv')
                   print("Pdf reading took" + str(time.time() - time_1) + " seconds")
                   time_2 = time.time()
                   score_df = get_current_song_comparison_scores(sample_scores_df, current_song_features)
                   print("Score comparison took " + str(time.time() - time_2) + " seconds")
                   print(score_df)
                   print("The program took " + str(time.time() - start_time) + " seconds")
                   print("The most similar song to " + sys.argv[2] + "is" + str(score_df.iloc[-1]))



