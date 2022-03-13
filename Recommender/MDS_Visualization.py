from sklearn.manifold import MDS
from matplotlib import pyplot as plt
import sklearn.datasets as dt
# import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from bokeh.io import curdoc, show
from bokeh.models import ColumnDataSource, Grid, ImageURL, LinearAxis, Plot, Range1d
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import sys

cid = 'c480b13ef81c4e6aa0ab0119636eabe5'
secret = '50826f24c12044448b906de50ac74742'
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

urls = []

def readAndCleanData():
    df = pd.read_pickle("Recommender/all_data.pkl")
    df = df[:100]
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
    # indices = list(range(len(urls)))
    # urlDict = dict(zip(indices, urls))  # Creates dictionary of index, url pair
    # urlDict = {str(k): v for k, v in urlDict.items()}  # Changes keys to string
    #
    # source = ColumnDataSource(urlDict)
    #
    xdr = Range1d(start=-10, end=10)
    ydr = Range1d(start=-10, end=10)

    plot = Plot(
        title=None, x_range=xdr, y_range=ydr, width=300, height=300,
        min_border=0, toolbar_location=None)

    for i in range(len(coordinates)):
        plot.add_glyph(ImageURL(url=[urls[i]], x=coordinates[i, 0], y=coordinates[i, 1], w=2, h=2, anchor="bottom_right"))

    xaxis = LinearAxis()
    plot.add_layout(xaxis, 'below')

    yaxis = LinearAxis()
    plot.add_layout(yaxis, 'left')

    plot.add_layout(Grid(dimension=0, ticker=xaxis.ticker))
    plot.add_layout(Grid(dimension=1, ticker=yaxis.ticker))

    curdoc().add_root(plot)

    show(plot)




    # # Just plot dots
    # fig = plt.figure(2, (10, 4))
    #
    # ax = fig.add_subplot(122)
    # plt.scatter(coordinates[:, 0], coordinates[:, 1])
    # plt.title('Embedding in 2D')
    # fig.subplots_adjust(wspace=.4, hspace=0.5)
    # plt.show()


cleanedData = readAndCleanData()
coordinates = getCoordinatesMDS(cleanedData)
plotAlbums(coordinates)

