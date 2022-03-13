from sklearn.manifold import MDS
import pandas as pd
import numpy as np
from bokeh.io import curdoc, show
from bokeh.models import ColumnDataSource, Grid, ImageURL, LinearAxis, Plot, Range1d
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn.preprocessing import normalize

cid = 'c480b13ef81c4e6aa0ab0119636eabe5'
secret = '50826f24c12044448b906de50ac74742'
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

urls = []

def readAndCleanData(N, b):
    df = pd.read_pickle("Recommender/all_data.pkl")
    df = df[:N]
    getURLS(df)

    df = df.drop(['URI'], axis=1)  # Drop URI

    # Normalize
    df[['key', 'loudness', 'tempo', 'duration_ms', 'time_signature']] = (df[
                                                                                              ['key', 'loudness', 'tempo',
                                                                                               'duration_ms',
                                                                                               'time_signature']] -
                                                                                          df[
                                                                                              ['key', 'loudness', 'tempo',
                                                                                               'duration_ms',
                                                                                               'time_signature']].min()) / (
                                                                                                     df[
                                                                                                         ['key', 'loudness',
                                                                                                          'tempo',
                                                                                                          'duration_ms',
                                                                                                          'time_signature']].max() -
                                                                                                     df[
                                                                                                         ['key', 'loudness',
                                                                                                          'tempo',
                                                                                                          'duration_ms',
                                                                                                          'time_signature']].min())

    albumValues = df.drop(
        ['Title', 'Artist', 'Descriptor Count', 'danceability', 'energy', 'key', 'loudness', 'mode',
         'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms',
         'time_signature'], axis=1)
    albumValuesCols = albumValues.columns.tolist()

    # Weighting of descriptors (greater b, less weight)
    df[albumValuesCols] = albumValues[albumValuesCols].apply(lambda x: x / b)

    df = df.to_numpy()
    df = np.delete(df, [0, 1, -1], 1)

    return df


def getCoordinatesMDS(df):

    # Do MDS
    mds = MDS(random_state=0)
    X_transform = mds.fit_transform(df)

    return X_transform


def getURLS(df):
    for uri in list(df['URI']):
        result = sp.album(uri)
        urls.append(result['images'][0]['url'])  # Append cover art URL to list of image URLS


def plotAlbums(coordinates):
    edge = max(coordinates.max(axis=0))  # Take max of x and y coordinates

    xdr = Range1d(start=-1 * edge, end=edge)
    ydr = Range1d(start=-1 * edge, end=edge)

    plot = Plot(
        title=None, x_range=xdr, y_range=ydr, width=3000, height=3000,
        min_border=0, toolbar_location=None)

    for i in range(len(coordinates)):
        plot.add_glyph(ImageURL(url=[urls[i]], x=coordinates[i, 0], y=coordinates[i, 1], w=.1, h=.1, anchor="bottom_right"))

    xaxis = LinearAxis()
    plot.add_layout(xaxis, 'below')

    yaxis = LinearAxis()
    plot.add_layout(yaxis, 'left')

    plot.add_layout(Grid(dimension=0, ticker=xaxis.ticker))
    plot.add_layout(Grid(dimension=1, ticker=yaxis.ticker))

    curdoc().add_root(plot)

    show(plot)


numAlbums = 500  # How many albums do we want to plot (Top numAlbums on RYM)
cleanedData = readAndCleanData(numAlbums, b=5)
coordinates = getCoordinatesMDS(cleanedData)
plotAlbums(coordinates)

