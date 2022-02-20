from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
from tabulate import tabulate

# Prints the 5 most similar albums
def printSimilar(titles, artists, uris, indices):
    similarAlbumsList = list(indices[90])
    print(distances[1:5])
    albums = []

    for i in range(len(similarAlbumsList)):
        albums.append([albumArtists[similarAlbumsList[i]], albumTitles[similarAlbumsList[i]], albumURIs[similarAlbumsList[i]]])

    print(tabulate(albums, headers=['Artist', 'Title', 'URI']))



# Using unsupervised KNN to get similar albums
albumsDataframe = pd.read_csv('Spotify API Connection/album_audio_feature_data.csv')
albumValues = albumsDataframe[['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature']]

albumTitles = list(albumsDataframe['Title'])  # List of album titles
albumArtists = list(albumsDataframe['Artist'])  # List of album artists
albumURIs = list(albumsDataframe['URI'])  # List of album artists

albumValues = albumValues.to_numpy(dtype=np.float32)  # np array of all audio metrics for albums
similarAlbums = NearestNeighbors(n_neighbors=6, algorithm='auto').fit(albumValues)
distances, indices = similarAlbums.kneighbors(albumValues)

printSimilar(albumTitles, albumArtists, albumURIs, indices)  # Print out similar albums
