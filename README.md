This program is intended to find for a given audio file the most similar audio (mostly song) from a given set (folder).
It widely implement 'librosa' functionality for music features extraction. Particulary, it analyzes each song for the following 
features: zeros_sum, rmse_mean, mean_tempo, mean_fundamental_frequency, mean_tempogram, nonzeros, mean_Xdb, mean_spectral, which serve,
i.e., by physical audio parameters. 
If you provide your own folder with the audio, the first program run may take up to 30 minutes to process the data. Each subsequent run 
will be much faster, since only one given audio would be analyzed. 

Usage

Install python 3.8 or later.

Copy the song file in the same directory where the file song_analyzer.py is located.

Run the program through this command:
python3 song_analyzer.py <songfile> <song samples directory(optional)> <distance computing method(optional)>

Unless you provide a directory path with song samples, a default sample directory ('songs') will be used.
Unless you provide the method name, the euclidean metrics will be used by default. Alternatively, the following
metrics can be applied: ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’, ‘cosine’, 
‘dice’, ‘euclidean’, ‘hamming’, ‘jaccard’, ‘jensenshannon’, ‘kulsinski’, ‘kulczynski1’, ‘mahalanobis’, ‘matching’, ‘minkowski’,
‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’.

Command examples:

python3 song_analyzer.py yesterday.mp3

or

python3 song_analyzer.py yesterday.mp3  my_song_samples  

or
python3 song_analyzer.py yesterday.mp3  my_song_samples cosine


The output will be the names of 3 most similar songs from the samples folder in the similarity descending order 
(the first one is more likely to be the closest one)



