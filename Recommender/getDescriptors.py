from sklearn.neighbors import NearestNeighbors
from skimage import io
from tabulate import tabulate
import matplotlib.pyplot as plt
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import numpy as np
import sys
import rymscraper
from rymscraper import RymUrl

#albumsDataframe = pd.read_csv('Spotify API Connection/album_audio_feature_data.csv')

albumsDataframe = pd.read_pickle("Recommender/albumsDataframe.pkl")
albumsDataframe['Title'] = albumsDataframe['Title'].replace(['Live 1966'],'The Bootleg Series Vol 4 Live 1966 The Royal Albert Hall Concert')
albumsDataframe['Artist'] = albumsDataframe['Artist'].replace(['The Notorious B.I.G.'],'the-notorious-b_i_g')
albumsDataframe['Artist'] = albumsDataframe['Artist'].replace(['山岡晃 [Akira Yamaoka]'],'山岡晃')
albumsDataframe['Title'] = albumsDataframe['Title'].replace(['Dream Letter'],'dream-letter-live-in-london-1968')
albumsDataframe['Title'] = albumsDataframe['Title'].replace(['Blues & Roots'],'Blues and Roots')
albumsDataframe['Artist'] = albumsDataframe['Artist'].replace(['青葉市子 [Ichiko Aoba]'],'青葉市子')
albumsDataframe['Title'] = albumsDataframe['Title'].replace(['Cera una volta il'],'Cera una volta il West')
albumsDataframe['Title'] = albumsDataframe['Title'].replace(['The Bootleg Volume 6'],'the-bootleg-series-vol-6-live-1964-concert-at-philharmonic-hall')
albumsDataframe['Artist'] = albumsDataframe['Artist'].replace(['Zappa'],'Zappa Mothers')
albumsDataframe['Artist'] = albumsDataframe['Artist'].replace(['Various Artists'],'岡部啓一-帆足圭吾')
albumsDataframe['Title'] = albumsDataframe['Title'].replace(['NieR:Automata'],'NieR_Automata')
albumsDataframe['Title'] = albumsDataframe['Title'].replace(['Vol. 4'], 'Vol 4')
albumsDataframe['Artist'] = albumsDataframe['Artist'].replace(['アトラスサウンドチーム'],'目黒将司')
albumsDataframe['Title'] = albumsDataframe['Title'].replace([':Blues'], '_Blues')


#albumsDataframe.to_pickle("Recommender/albumsDataframe.pkl")







network = rymscraper.RymNetwork()

# Dictionary that holds all descriptors and how many albums have each
descriptors = {'melancholic': 170, 'anxious': 116, 'futuristic': 32, 'male vocals': 393, 'existential': 90, 'alienation': 80, 'atmospheric': 183, 'lonely': 93, 'cold': 57, 'pessimistic': 50, 'introspective': 146, 'depressive': 56, 'longing': 93, 'dense': 105, 'sarcastic': 54, 'serious': 44, 'urban': 96, 'progressive': 134, 'passionate': 251, 'concept album': 63, 'bittersweet': 158, 'meditative': 33, 'epic': 105, 'complex': 128, 'sentimental': 90, 'melodic': 260, 'fantasy': 33, 'philosophical': 78, 'surreal': 103, 'poetic': 149, 'abstract': 60, 'technical': 97, 'improvisation': 61, 'avant-garde': 76, 'psychedelic': 134, 'medieval': 6, 'political': 47, 'sombre': 86, 'cryptic': 81, 'winter': 22, 'hypnotic': 97, 'mysterious': 93, 'dark': 125, 'nocturnal': 135, 'rhythmic': 210, 'apocalyptic': 39, 'ethereal': 49, 'apathetic': 14, 'eclectic': 111, 'conscious': 66, 'protest': 27, 'religious': 19, 'spiritual': 60, 'Christian': 16, 'uplifting': 96, 'noisy': 55, 'romantic': 70, 'love': 102, 'female vocals': 61, 'lush': 108, 'warm': 159, 'Wall of Sound': 8, 'sensual': 24, 'sexual': 51, 'androgynous vocals': 10, 'soothing': 57, 'mellow': 102, 'space': 17, 'calm': 34, 'summer': 47, 'happy': 25, 'medley': 1, 'quirky': 75, 'playful': 131, 'optimistic': 32, 'energetic': 199, 'suite': 22, 'drugs': 61, 'raw': 102, 'nihilistic': 33, 'rebellious': 67, 'hedonistic': 33, 'deadpan': 12, 'dissonant': 29, 'lo-fi': 26, 'science fiction': 25, 'anthemic': 52, 'rock opera': 5, 'triumphant': 32, 'LGBT': 15, 'sampling': 54, 'humorous': 49, 'boastful': 34, 'crime': 22, 'repetitive': 45, 'manic': 49, 'tribal': 12, 'instrumental': 72, 'suspenseful': 51, 'acoustic': 65, 'death': 59, 'autumn': 45, 'polyphonic': 17, 'violence': 31, 'alcohol': 15, 'aquatic': 12, 'heavy': 97, 'war': 17, 'ominous': 75, 'funereal': 11, 'vocal group': 4, 'orchestral': 17, 'chamber music': 3, 'history': 8, 'aggressive': 64, 'vulgar': 18, 'uncommon time signatures': 59, 'pastoral': 30, 'soft': 34, 'peaceful': 29, 'minimalistic': 15, 'sparse': 14, 'monologue': 2, 'sad': 43, 'rain': 7, 'breakup': 29, 'self-hatred': 21, 'satirical': 20, 'angry': 51, 'misanthropic': 35, 'chaotic': 42, 'folklore': 4, 'nature': 17, 'seasonal': 4, 'spring': 20, 'forest': 11, 'scary': 17, 'ballad': 7, 'suicide': 12, 'disturbing': 28, 'lethargic': 16, 'choral': 6, 'infernal': 17, 'mechanical': 18, 'occult': 17, 'pagan': 8, 'tropical': 16, '': 10, 'anti-religious': 10, 'hateful': 8, 'party': 12, 'mashup': 1, 'desert': 9, 'martial': 2, 'ritualistic': 14, 'mythology': 5, 'satanic': 9, 'parody': 1, 'paranormal': 4, 'Halloween': 3, 'anarchism': 1, 'natural': 2, 'lyrics': 1, 'waltz': 1, 'Islamic': 1, 'atonal': 1, 'sports': 2, 'oratorio': 1}
allAlbumDescriptorValues = []

# Prints the 5 most similar albums
def getAlbumDescriptors(artist, albumTitle):
    try:
        album = str(artist) + " - " + str(albumTitle)
        album_infos = network.get_album_infos(name=album)['Descriptors']
    except IndexError:
        artist = str(artist).replace(' /', '').lower()
        artist = str(artist).replace('&', 'and').lower()
        artist = str(artist).replace(' ', '-').lower()
        artist = str(artist).replace(',', '').lower()
        title = str(albumTitle).replace(' ', '-').lower()
        title = str(title).replace('\'', '').lower()
        title = str(title).replace('(', '').lower()
        title = str(title).replace(')', '').lower()
        title = str(title).replace('%', '').lower()
        title = str(title).replace('&', 'and').lower()
        title = str(title).replace(':', '').lower()
        title = str(title).replace('é', 'e').lower()
        title = str(title).replace('à', 'a').lower()
        url = "https://rateyourmusic.com/release/album/" + artist + "/" + title + "/"
        print(url)
        album_infos = network.get_album_infos(url=url)['Descriptors']

    albumDescriptorsList = [x.strip() for x in album_infos.split(',')]
    return albumDescriptorsList
    #print(albumDescriptorsList)

# Plot similar albums
def getAllDescriptors(listOfAlbums):
    listOfAlbums = listOfAlbums.reset_index()

    startFrom = 0

    for index, album in listOfAlbums.iterrows():
        print(index, album['Artist'], album['Title'])

        if index >= startFrom:
            artist = album['Artist']
            albumTitle = album['Title']
            thisAlbumDescriptors = getAlbumDescriptors(artist, albumTitle)

            print(index, albumTitle, artist, thisAlbumDescriptors)

            for descriptor in thisAlbumDescriptors:
                if descriptor not in descriptors:
                    descriptors[descriptor] = 1
                else:
                    descriptors[descriptor] += 1  # to get an idea of most popular descriptors

            print("\nAll Descriptors:", descriptors, "\n\n")

def getDescriptorVectors(listOfAlbums):
    listOfAlbums = listOfAlbums.reset_index()

    startFrom = 400
    end = 510
    descriptorVal = 63  # Initializes first descriptor with weight 1.5, and last descriptor minimum 0.5

    for index, album in listOfAlbums.iterrows():
        if index >= startFrom and index < end:
            artist = album['Artist']
            albumTitle = album['Title']
            thisAlbumDescriptors = getAlbumDescriptors(artist, albumTitle)
            albumDescVector = descriptors  # make a copy of overall descriptors (will set 0 or 1 to each value of descriptor)

            thisAlbumDict = {k: v for v, k in enumerate(thisAlbumDescriptors)}  # Descriptor:Index pair dictionary for quick lookup

            for descriptor in albumDescVector.keys():
                if descriptor in thisAlbumDescriptors:
                    albumDescVector[descriptor] = (descriptorVal - thisAlbumDict[descriptor]) / 42  # 42 is max number of descriptors for an album
                else:
                    albumDescVector[descriptor] = 0

            listValues = list(albumDescVector.values())
            listValues.append(len(thisAlbumDescriptors))  # Append number of descriptors in album for later use
            allAlbumDescriptorValues.append(listValues)

            print(index, albumTitle, artist)
            #print(allAlbumDescriptorValues)


    columnNames = list(descriptors.keys())
    columnNames.append("Descriptor Count")
    finalDescriptorDataframe = pd.DataFrame(allAlbumDescriptorValues, columns=columnNames)

    print(finalDescriptorDataframe)

    finalDescriptorDataframe.to_pickle("Recommender/descriptors_data_priori_400-503.pkl")


def combineDataframes():
    dfOne = pd.read_pickle("Recommender/descriptors_data_priori_0-99.pkl")
    dfTwo = pd.read_pickle("Recommender/descriptors_data_priori_100-199.pkl")
    dfThree = pd.read_pickle("Recommender/descriptors_data_priori_200-249.pkl")
    dfFour = pd.read_pickle("Recommender/descriptors_data_priori_250-399.pkl")
    dfFive = pd.read_pickle("Recommender/descriptors_data_priori_400-503.pkl")

    allAlbumsDescriptorData = pd.concat([dfOne, dfTwo, dfThree, dfFour, dfFive], axis=0)
    allAlbumsDescriptorData.reset_index()
    print(allAlbumsDescriptorData)

    allAlbumsDescriptorData.to_pickle("Recommender/all_albums_priori_descriptors.pkl")

#getAllDescriptors(albumsDataframe)


#getDescriptorVectors(albumsDataframe)

#combineDataframes()




# Next step: for each descriptor start at len(albumDescriptors) and decrement for each additional descriptor so that some descriptors are higher than others for each album
#getDescriptorVectors(albumsDataframe)




