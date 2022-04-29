import numpy as np
import time
import pandas as pd
from os import listdir
import audio_features_extractor
from os.path import isfile, join
from scipy.spatial.distance import cdist
from os.path import exists
import argparse
import warnings



parser = argparse.ArgumentParser()
parser.add_argument("-e", "--explicit", action="store_true",
                    help="explicit output")
args = parser.parse_known_args()

if (str(args[0]) != "Namespace(explicit=True)"):
    warnings.filterwarnings("ignore")
#print(args[0]) # to print out the flag value
# print(args[1][2]) # to print the method argument passed



def compare_two_songs(song1_data, song2_data, method):
    # print("printing songs data")
    # print(song1_data)
    # print(song2_data)
    similarity = cdist([song1_data], [song2_data], method)
    return np.average(similarity)

def analyze_all_songs(song_samples_dir):
    df_songs_features = pd.DataFrame()

    print(df_songs_features.shape)
    for songfile in listdir(song_samples_dir):
        if isfile(join(song_samples_dir, songfile)):
            temp_df = audio_features_extractor.get_song_features(song_samples_dir + '/' + songfile, str(args[0]))

            df_songs_features = pd.concat([df_songs_features, temp_df], ignore_index=True)

    df_songs_features.to_csv(song_samples_dir + "_features.csv")



def get_current_song_comparison_scores(sample_scores_df, current_song_features):
    song_comparison_scores = pd.DataFrame({'song_name':[], 'score':[]} )

    for index, row in sample_scores_df.iterrows():
        # print("current_song_features")
        # print(current_song_features.values.tolist()[0])

        current_score = compare_two_songs(row[2:9].values.tolist(), current_song_features.values.tolist()[0][1:8], args[1][2])
        df_with_cuurent_score = pd.DataFrame({'song_name': row['song_name'], 'score': [current_score]})

        song_comparison_scores = pd.concat([song_comparison_scores, df_with_cuurent_score], ignore_index=True)

    song_comparison_scores = song_comparison_scores.sort_values(['score'], ascending=[False])
    return song_comparison_scores




if __name__ == '__main__':
       start_time = time.time()
       usage = """Incorrect command arguments! Usage:
              python3 song_analyzer.py <songfile> <song samples directory(optional)> <distance computing method(optional)>
              
              or (for detailed output)
                python3 song_analyzer.py -e <songfile> <song samples directory(optional)> <distance computing method(optional)>
              """
       if (len(args) < 2 or len(args) > 4):
           print(usage)
       else:

           song_samples_dir = 'songs'
           if (args[1][1] is not None):
               song_samples_dir = args[1][1]
               if (not exists (song_samples_dir + '_features.csv')):
                   analyze_all_songs(song_samples_dir)
                   print("CREATING DATASET! IT MAY TAKE A WHILE")
                   print("THE DATASET IS CREATED!!!")
                   current_song_features = audio_features_extractor.get_song_features(args[1][0], str(args[0]))
                   sample_scores_df = pd.read_csv(song_samples_dir + '_features.csv')
                   score_df = get_current_song_comparison_scores(sample_scores_df, current_song_features)
                   if (str(args[0]) == "Namespace(explicit=True)"):
                       print(score_df)
                       print("The program took " + str(time.time() - start_time) + " seconds")
                       print("The most similar song to " + args[1][0] + "is" + str(score_df.iloc[-1]))
                   else:
                       print(str(score_df.iloc[-1]["song_name"]))

               else:
                   if (str(args[0]) == "Namespace(explicit=True)"):
                       print("USING THE DEFAULT DATASET!!!")
                   current_song_features = audio_features_extractor.get_song_features(args[1][0], str(args[0]))
                   print("current_song_features")
                   print(current_song_features)
                   #print("Current song analysis took " + str(time.time() - start_time) + " seconds")
                   time_1 = time.time()
                   sample_scores_df = pd.read_csv(song_samples_dir + '_features.csv')
                   #print("Pdf reading took" + str(time.time() - time_1) + " seconds")
                   time_2 = time.time()
                   score_df = get_current_song_comparison_scores(sample_scores_df, current_song_features)
                   #print("Score comparison took " + str(time.time() - time_2) + " seconds")
                   if (str(args[0]) == "Namespace(explicit=True)"):
                       print(score_df)
                       print("The program took " + str(time.time() - start_time) + " seconds")
                       print("The most similar song to " + args[1][0] + "is" + str(score_df.iloc[-1]))
                       print("The most similar song to " + args[1][0] + "is" + str(score_df.iloc[-2]))
                       print("The most similar song to " + args[1][0] + "is" + str(score_df.iloc[-3]))
                   else:
                       print(str(score_df.iloc[-1]["song_name"]) + ", " + str(score_df.iloc[-2]["song_name"]) + ", " + str(score_df.iloc[-3]["song_name"]))




