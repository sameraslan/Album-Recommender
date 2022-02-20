from sklearn.neighbors import NearestNeighbors
from skimage import io
from tabulate import tabulate
import matplotlib.pyplot as plt
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import numpy as np

cid = 'c480b13ef81c4e6aa0ab0119636eabe5'
secret = '50826f24c12044448b906de50ac74742'
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Prints the 5 most similar albums
def printSimilar(titles, artists, uris, indices):
    similarAlbumsList = list(indices[90])  # Specified Album to find similar albums to
    albums = []
    recommendedUris = []
    recommendedTitles = []

    for i in range(len(similarAlbumsList)):
        albums.append([artists[similarAlbumsList[i]], titles[similarAlbumsList[i]], uris[similarAlbumsList[i]]])
        recommendedUris.append(uris[similarAlbumsList[i]])
        recommendedTitles.append(titles[similarAlbumsList[i]])

    print(tabulate(albums, headers=['Artist', 'Title', 'URI']))
    visualizeAlbums(recommendedUris, recommendedTitles)

def visualizeAlbums(uris, titles):
    urls = []  # List of cover art URLS

    for uri in uris:
        result = sp.album(uri)
        urls.append(result['images'][0]['url'])  # Append cover art URL to list of image URLS

    plt.figure(figsize=(15, int(0.625 * len(uris))), facecolor='#ffeba3')
    columns = len(urls)

    for i, url in enumerate(urls):
        plt.subplot(int(len(urls) / columns), columns, i + 1)

        image = io.imread(url)
        plt.imshow(image)
        plt.xticks([])
        plt.yticks([])
        s = ''
        #plt.xlabel(s.join(playlist_df['track_name'].values[i].split(' ')[:4]), fontsize=10, fontweight='bold')
        plt.tight_layout(h_pad=1.2, w_pad=0)
        plt.subplots_adjust(wspace=None, hspace=None)

    plt.show()



# Using unsupervised KNN to get similar albums (euclidean distance)
albumsDataframe = pd.read_csv('Spotify API Connection/album_audio_feature_data.csv')
albumValues = albumsDataframe[['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature']]

albumTitles = list(albumsDataframe['Title'])  # List of album titles
albumArtists = list(albumsDataframe['Artist'])  # List of album artists
albumURIs = list(albumsDataframe['URI'])  # List of album artists

albumValues = albumValues.to_numpy(dtype=np.float32)  # np array of all audio metrics for albums
similarAlbums = NearestNeighbors(n_neighbors=6, algorithm='auto').fit(albumValues)
distances, indices = similarAlbums.kneighbors(albumValues)

printSimilar(albumTitles, albumArtists, albumURIs, indices)  # Print out similar albums
