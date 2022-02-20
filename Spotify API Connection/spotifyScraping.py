import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import numpy as np

cid = 'c480b13ef81c4e6aa0ab0119636eabe5'
secret = '50826f24c12044448b906de50ac74742'
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

artist_uri = '5LhTec3c7dcqBvpLRWbMcf'
track_uri = 'spotify:track:36apwMphkcaS63LY3JJMPh'
#album_uri = 'spotify:album:7GOdEIOvr41lvxDK7bvPrI'

all_album_features = []
all_album_names = []
all_album_artists = []
all_album_uri = []

df = pd.read_csv("RYMScraper/Scraped Data/Above1kRatings.csv")

#Albums deleted (not in spotify) (index iloc based)
df = df.drop([54])  # King Crimson,The Great Deceiver: Live 1973-1974
df = df.drop([187])  # Joanna Newsom,Ys
df = df.drop([225])  # Electric Masada, At the Mountains of Madness
df = df.drop([239])  # Kraftwerk,Die Mensch-Maschine
df = df.drop([241])  # Shiro Sagisu,The End of Evangelion
df = df.drop([252])  # Les Rallizes dénudés,'77 Live
df = df.drop([310])  # David Wise,Donkey Kong Country 2: Diddy's Kong Quest
df = df.drop([323])  # 田中宏和 [Hirokazu Tanaka] & 鈴木慶一 [Keiichi Suzuki],Mother 2: ギーグの逆襲
df = df.drop([324])  # 近藤浩治 [Koji Kondo],ゼルダの伝説: ムジュラの仮面 (The Legend of Zelda: Majora's Mask)
df = df.drop([326])  # Staatsorchester Stuttgart,Tabula rasa
df = df.drop([375])  # Mario Galaxy Orchestra,Super Mario Galaxy
df = df.drop([381])  # Joanna Newsom,Have One on Me
df = df.drop([383])  # Boris,Flood
df = df.drop([384])  # Organized Konfusion,Stress: The Extinction Agenda
df = df.drop([390])  # 三宅優 [Yu Miyake],塊魂サウンドトラック「塊フォルテッシモ魂」 (Katamari Damacy Soundtrack: Katamari Fortissimo Damacy)

#Albums deleted (not in spotify) (label loc based)
df = df.drop(431)  # Dinosaur   You're Living All Over Me
df = df.drop(434)  # Death Grips  Jenny Death: The Powers That B Disc 2
df = df.drop(436)  # 久石譲 [Joe Hisaishi] もののけ姫 (Mononoke-hime)
df = df.drop(451)  # Pink Floyd  Is There Anybody Out There? The Wall Live 1980-81

sp.trace = False

# find album by name
#i and j are ranges of rows in df to search for albums
i = 0
j = 2


# get the first album uri
df = df.loc[i:j]
df = df[['Artist', 'Album']]


for index, row in df.iterrows():
    album_uri = ''
    #Specifies artist as well by concatinating in order to improve search accuracy of album
    albumName = str(row['Album']) + " " + str(row['Artist'])
    print(albumName) #(for testing)
    # Catch Error due to too specific artist or album name (spotify RYM mismatch)
    # Increases chance of finding album in spotify
    try:
        results = sp.search(q="album:" + albumName, type="album")
        album_uri = results['albums']['items'][0]['uri']
    except IndexError:
        try:
            albumName = str(row['Album']) + " " + str(row['Artist'])[:5]
            results = sp.search(q="album:" + albumName, type="album")
            album_uri = results['albums']['items'][0]['uri']
        except IndexError:
            pass

    if album_uri != '':
        album_title = sp.album(album_uri)
        print(str(i), str(album_title['name']))
        i += 1

        # get album tracks and testing to get accurate results
        # Retrieve audio_features for each track
        # album = 'Kid A Radiohead'
        # results = sp.search(q="album:" + album, type="album")
        # album_uri = results['albums']['items'][0]['uri']

        tracks = sp.album_tracks(album_uri)
        count = 0
        track_features = []  # Store features for each track
        for track in tracks['items']:
            #print(track['name'], track['uri'])
            track_uri = track['uri']
            results = sp.audio_features(track_uri)
            track_features.append(list(results[0].values()))

        album_features = np.array(track_features)

        album_features = np.delete(album_features, list(range(11, 16)), 1).astype(np.float32)  # Remove non-numerical items and cast to float
        album_features = np.mean(album_features, axis=0)  # Take column wise mean for overall album audio features
        print(album_features)
        all_album_features.append(list(album_features))
        all_album_names.append(str(row['Album']))
        all_album_artists.append(str(row['Artist']))
        all_album_uri.append(album_uri)

        if index == 0:
            label_names = np.array(list(results[0].keys())[0:11] + list(results[0].keys())[16:])  # ['danceability', 'energy', 'key', 'loudness',...


album_data_dataframe = pd.DataFrame(all_album_features, columns=label_names)
album_data_dataframe.insert(0, "Artist", all_album_artists)
album_data_dataframe.insert(0, "Title", all_album_names)
album_data_dataframe["URI"] = all_album_uri
print(album_data_dataframe)