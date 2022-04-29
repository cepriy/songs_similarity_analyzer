import librosa
import numpy as np
import time
import pandas as pd

start_position = 200000
end_position = 6500000

def get_song_features(songfile, mode):
    start_song_analysis = time.time()
    x, sr = librosa.load(songfile,
                         sr=None)  # loads an audio file and decodes it into a 1-dimensional array which is a time series x , and sr is a sampling rate of x
    #Keep in mind that a song on avarage is of about a 7-8 million numpy array size.
    # print("Printing duration of " + str(songfile))
    # print(len(x))

    if (len(x) > end_position):
        x = x[start_position: end_position]
    else:
        if (mode == "Namespace(explicit=True)"):
            print("The song " + str(songfile) + " is too short; the result may be biased")

    #print("Mp3 to nmpy array took" + str(time.time() - start_song_analysis) + " seconds")
    time_stamp_2 = time.time()
    #nonzero = np.count_nonzero(x)/(len(x)) # proportional to the duration of non-silent fragments, reduced by 10000000 times
                                        # Nonzeros allow measuring the percussion characteristics
    #print("Nonzero calculating took " + str(time.time() - time_stamp_2) + " seconds")
    time_stamp_3 = time.time()

    zeros = librosa.zero_crossings(x) # Zeros measure the silence
    zeros_sum = np.sum(zeros)/len(x)*10 #division for normalizing POSSIBLY MUTIPLY BY TEN
    #print("Zeros took" + str(time.time() - time_stamp_3) + " seconds")
    time_stamp_4 = time.time()
    rmse = librosa.feature.rms(y=x)[0] # rmse corresponds to the energy level of an audio
    rmse_mean = np.mean(rmse)
    #print("Rmse took" + str(time.time() - time_stamp_4) + " seconds")
    time_stamp_5 = time.time()
    f0 = librosa.yin(x, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')) #possible bottle neck
    mean_fundamental_frequency = np.mean(f0) / 1000 * 2 #to this feature provide more weight
    #times = librosa.times_like(f0)


    #print("F0 took" + str(time.time() - time_stamp_5) + " seconds")
    time_stamp_6 = time.time()

    #print("Mean fundamental" + str(time.time() - time_stamp_6) + " seconds")
    time_stamp_7 = time.time()

    mean_tempo = np.mean(librosa.beat.tempo(y=x)[0])/1000
    #print("Mean tempo took" + str(time.time() - time_stamp_7) + " seconds")
    time_stamp_8 = time.time()
    hop_length = 512
    y = x
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length) #Spectral flux is a measure of how quickly the power spectrum of a signal is changing
    #print("Oenv took" + str(time.time() - time_stamp_8) + " seconds")
    time_stamp_9 = time.time()
    tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr, hop_length=hop_length)
    #print("Tempogram took" + str(time.time() - time_stamp_9) + " seconds")

    mean_tempogram = np.mean(tempogram)
    time_stamp_10 = time.time()
    X = librosa.stft(x)
    Xdb = librosa.amplitude_to_db(abs(X))
    mean_Xdb = np.mean(Xdb)/100   #It is reasonable to play around with median, mode of some features
    #print("Xdb" + str(time.time() - time_stamp_10) + " seconds")
    time_stamp_11 = time.time()

    spectral_centroids = librosa.feature.spectral_centroid(y=x, sr=sr)[0]
    mean_spectral = np.mean(spectral_centroids)/10000
    #print("Mean spectral took" + str(time.time() - time_stamp_11) + " seconds")

    return pd.DataFrame({'song_name': songfile,
                  'zeros_sum': [zeros_sum],
                  'rmse_mean': [rmse_mean],
                  'mean_tempo': [mean_tempo],
                  'mean_fund_freq': [mean_fundamental_frequency],
                  'mean_tempogram': [mean_tempogram],
                  'mean_xDB': [ mean_Xdb],
                  'mean_spectral': [mean_spectral]})



