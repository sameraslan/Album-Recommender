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
from rymscraper import rymscraper, RymUrl

#albumsDataframe = pd.read_csv('Spotify API Connection/album_audio_feature_data.csv')

# albumsDataframe = pd.read_pickle("Recommender/albumsDataframe.pkl")
# albumsDataframe['Title'] = albumsDataframe['Title'].replace(['Live 1966'],'The Bootleg Series Vol 4 Live 1966 The Royal Albert Hall Concert')
# albumsDataframe['Artist'] = albumsDataframe['Artist'].replace(['The Notorious B.I.G.'],'the-notorious-b_i_g')
# albumsDataframe['Artist'] = albumsDataframe['Artist'].replace(['山岡晃 [Akira Yamaoka]'],'山岡晃')
# albumsDataframe['Title'] = albumsDataframe['Title'].replace(['Dream Letter'],'dream-letter-live-in-london-1968')
# albumsDataframe['Title'] = albumsDataframe['Title'].replace(['Blues & Roots'],'Blues and Roots')
# albumsDataframe['Artist'] = albumsDataframe['Artist'].replace(['青葉市子 [Ichiko Aoba]'],'青葉市子')
# albumsDataframe['Title'] = albumsDataframe['Title'].replace(['Cera una volta il'],'Cera una volta il West')
# albumsDataframe['Title'] = albumsDataframe['Title'].replace(['The Bootleg Volume 6'],'the-bootleg-series-vol-6-live-1964-concert-at-philharmonic-hall')
# albumsDataframe['Artist'] = albumsDataframe['Artist'].replace(['Zappa'],'Zappa Mothers')
# albumsDataframe['Artist'] = albumsDataframe['Artist'].replace(['Various Artists'],'岡部啓一-帆足圭吾')
# albumsDataframe['Title'] = albumsDataframe['Title'].replace(['NieR:Automata'],'NieR_Automata')
# albumsDataframe['Title'] = albumsDataframe['Title'].replace(['Vol. 4'], 'Vol 4')
# albumsDataframe['Artist'] = albumsDataframe['Artist'].replace(['アトラスサウンドチーム'],'目黒将司')
# albumsDataframe['Title'] = albumsDataframe['Title'].replace([':Blues'], '_Blues')

#albumsDataframe.to_pickle("Recommender/albumsDataframe.pkl")

network = rymscraper.RymNetwork()

# Dictionary that holds all descriptors and how many albums have each
oldDescriptors = {'melancholic': 170, 'anxious': 116, 'futuristic': 32, 'male vocals': 393, 'existential': 90, 'alienation': 80, 'atmospheric': 183, 'lonely': 93, 'cold': 57, 'pessimistic': 50, 'introspective': 146, 'depressive': 56, 'longing': 93, 'dense': 105, 'sarcastic': 54, 'serious': 44, 'urban': 96, 'progressive': 134, 'passionate': 251, 'concept album': 63, 'bittersweet': 158, 'meditative': 33, 'epic': 105, 'complex': 128, 'sentimental': 90, 'melodic': 260, 'fantasy': 33, 'philosophical': 78, 'surreal': 103, 'poetic': 149, 'abstract': 60, 'technical': 97, 'improvisation': 61, 'avant-garde': 76, 'psychedelic': 134, 'medieval': 6, 'political': 47, 'sombre': 86, 'cryptic': 81, 'winter': 22, 'hypnotic': 97, 'mysterious': 93, 'dark': 125, 'nocturnal': 135, 'rhythmic': 210, 'apocalyptic': 39, 'ethereal': 49, 'apathetic': 14, 'eclectic': 111, 'conscious': 66, 'protest': 27, 'religious': 19, 'spiritual': 60, 'Christian': 16, 'uplifting': 96, 'noisy': 55, 'romantic': 70, 'love': 102, 'female vocals': 61, 'lush': 108, 'warm': 159, 'Wall of Sound': 8, 'sensual': 24, 'sexual': 51, 'androgynous vocals': 10, 'soothing': 57, 'mellow': 102, 'space': 17, 'calm': 34, 'summer': 47, 'happy': 25, 'medley': 1, 'quirky': 75, 'playful': 131, 'optimistic': 32, 'energetic': 199, 'suite': 22, 'drugs': 61, 'raw': 102, 'nihilistic': 33, 'rebellious': 67, 'hedonistic': 33, 'deadpan': 12, 'dissonant': 29, 'lo-fi': 26, 'science fiction': 25, 'anthemic': 52, 'rock opera': 5, 'triumphant': 32, 'LGBT': 15, 'sampling': 54, 'humorous': 49, 'boastful': 34, 'crime': 22, 'repetitive': 45, 'manic': 49, 'tribal': 12, 'instrumental': 72, 'suspenseful': 51, 'acoustic': 65, 'death': 59, 'autumn': 45, 'polyphonic': 17, 'violence': 31, 'alcohol': 15, 'aquatic': 12, 'heavy': 97, 'war': 17, 'ominous': 75, 'funereal': 11, 'vocal group': 4, 'orchestral': 17, 'chamber music': 3, 'history': 8, 'aggressive': 64, 'vulgar': 18, 'uncommon time signatures': 59, 'pastoral': 30, 'soft': 34, 'peaceful': 29, 'minimalistic': 15, 'sparse': 14, 'monologue': 2, 'sad': 43, 'rain': 7, 'breakup': 29, 'self-hatred': 21, 'satirical': 20, 'angry': 51, 'misanthropic': 35, 'chaotic': 42, 'folklore': 4, 'nature': 17, 'seasonal': 4, 'spring': 20, 'forest': 11, 'scary': 17, 'ballad': 7, 'suicide': 12, 'disturbing': 28, 'lethargic': 16, 'choral': 6, 'infernal': 17, 'mechanical': 18, 'occult': 17, 'pagan': 8, 'tropical': 16, '': 10, 'anti-religious': 10, 'hateful': 8, 'party': 12, 'mashup': 1, 'desert': 9, 'martial': 2, 'ritualistic': 14, 'mythology': 5, 'satanic': 9, 'parody': 1, 'paranormal': 4, 'Halloween': 3, 'anarchism': 1, 'natural': 2, 'lyrics': 1, 'waltz': 1, 'Islamic': 1, 'atonal': 1, 'sports': 2, 'oratorio': 1}
# From first 962 albums
descriptors = {'melancholic': 271, 'anxious': 167, 'futuristic': 55, 'male vocals': 650, 'existential': 120, 'alienation': 110, 'atmospheric': 311, 'lonely': 144, 'cold': 84, 'pessimistic': 70, 'introspective': 216, 'depressive': 84, 'longing': 147, 'dense': 169, 'sarcastic': 76, 'serious': 60, 'urban': 148, 'progressive': 204, 'passionate': 408, 'concept album': 104, 'bittersweet': 253, 'meditative': 52, 'epic': 181, 'complex': 203, 'sentimental': 149, 'melodic': 423, 'political': 72, 'conscious': 106, 'poetic': 214, 'protest': 37, 'eclectic': 177, 'religious': 28, 'spiritual': 95, 'rhythmic': 318, 'Christian': 21, 'uplifting': 145, 'fantasy': 61, 'philosophical': 94, 'surreal': 161, 'abstract': 91, 'technical': 156, 'improvisation': 125, 'avant-garde': 125, 'psychedelic': 201, 'medieval': 12, 'sombre': 141, 'cryptic': 112, 'winter': 36, 'hypnotic': 149, 'mysterious': 127, 'dark': 203, 'nocturnal': 208, 'apocalyptic': 74, 'ethereal': 90, 'apathetic': 21, 'noisy': 100, 'romantic': 111, 'love': 150, 'female vocals': 104, 'lush': 184, 'Wall of Sound': 17, 'warm': 249, 'sensual': 40, 'sexual': 78, 'androgynous vocals': 14, 'soothing': 89, 'mellow': 161, 'space': 29, 'calm': 61, 'summer': 75, 'happy': 40, 'medley': 4, 'quirky': 121, 'playful': 213, 'optimistic': 40, 'energetic': 353, 'suite': 35, 'sampling': 94, 'humorous': 84, 'boastful': 54, 'drugs': 89, 'deadpan': 22, 'crime': 36, 'raw': 176, 'nihilistic': 46, 'rebellious': 105, 'hedonistic': 49, 'dissonant': 59, 'lo-fi': 40, 'science fiction': 43, 'anthemic': 88, 'rock opera': 6, 'triumphant': 59, 'LGBT': 22, 'death': 91, 'autumn': 65, 'polyphonic': 27, 'repetitive': 78, 'manic': 92, 'tribal': 21, 'instrumental': 163, 'suspenseful': 80, 'acoustic': 133, 'violence': 51, 'alcohol': 27, 'aquatic': 22, 'heavy': 163, 'war': 30, 'ominous': 125, 'funereal': 19, 'chamber music': 11, 'vocal group': 4, 'orchestral': 44, 'history': 13, 'vulgar': 37, 'aggressive': 119, 'uncommon time signatures': 86, 'monologue': 4, 'pastoral': 56, 'soft': 59, 'peaceful': 49, 'minimalistic': 30, 'sparse': 29, 'sad': 64, 'rain': 16, 'breakup': 45, 'misanthropic': 58, 'satirical': 26, 'angry': 76, 'self-hatred': 27, 'chaotic': 79, 'folklore': 8, 'nature': 35, 'seasonal': 6, 'spring': 27, 'forest': 20, 'choral': 14, 'scary': 25, 'lethargic': 30, 'ballad': 8, 'disturbing': 43, 'mechanical': 30, 'suicide': 16, 'infernal': 30, 'occult': 28, 'pagan': 8, 'tropical': 23, 'party': 22, 'anti-religious': 15, 'hateful': 13, 'mashup': 1, 'ritualistic': 23, 'desert': 15, 'martial': 12, 'mythology': 9, 'natural': 6, 'satanic': 11, 'skit': 2, 'parody': 1, 'paranormal': 7, 'Halloween': 6, 'anarchism': 3, 'atonal': 6, 'Islamic': 3, 'lyrics': 1, 'waltz': 1, 'jingle': 1, 'opera': 1, 'symphony': 7, 'sports': 2, 'fairy tale': 1, 'oratorio': 1, 'ensemble': 5, 'string quartet': 2, 'ideology': 2, 'educational': 1}



allAlbumDescriptorValues = []
notFound = []

# Fixes album or artist name to suit url
def fixStringForWebsite(string):
    string = string.replace(' /', '').lower()
    string = string.replace('&', 'and').lower()
    string = string.replace(' ', '-').lower()
    string = string.replace(',', '').lower()
    string = string.replace('\'', '').lower()
    string = string.replace('(', '').lower()
    string = string.replace(')', '').lower()
    string = string.replace('[', '').lower()
    string = string.replace(']', '').lower()
    string = string.replace('%', '').lower()
    string = string.replace(':', '').lower()
    string = string.replace('é', 'e').lower()
    string = string.replace('à', 'a').lower()
    string = string.replace('.', '').lower()

    return string

# Prints the 5 most similar albums
def getAlbumDescriptors(artist, albumTitle):
    album_infos = []

    try:
        album = str(artist) + " - " + str(albumTitle)
        album_infos = network.get_album_infos(name=album)['Descriptors']
    except IndexError:
        try:
            artist = fixStringForWebsite(str(artist))
            title = fixStringForWebsite(str(albumTitle))

            url = "https://rateyourmusic.com/release/album/" + artist + "/" + title + "/"
            print(url)
            album_infos = network.get_album_infos(url=url)['Descriptors']
        except IndexError:
            notFound.append([artist, albumTitle])
        except AttributeError:
            notFound.append([artist, albumTitle])
    except AttributeError:
        notFound.append([artist, albumTitle])
    except TypeError:
        notFound.append([artist, albumTitle])

    if not len(album_infos) == 0:
        albumDescriptorsList = [x.strip() for x in album_infos.split(',')]
        return albumDescriptorsList

    return album_infos
    #print(albumDescriptorsList)

# Plot similar albums
def getAllDescriptors(listOfAlbums):
    listOfAlbums = listOfAlbums.reset_index()

    startFrom = 428

    for index, album in listOfAlbums.iterrows():
        print(index, album['Artist'], album['Album'])

        if index >= startFrom:
            artist = album['Artist']
            albumTitle = album['Album']
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
            albumTitle = album['Album']
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



#getDescriptorVectors(albumsDataframe)

#combineDataframes()

listOfAlbums = pd.read_pickle("rymscraper-master/Scraped Data/top5000records.pkl")
#print(listOfAlbums)
getAllDescriptors(listOfAlbums)
print(notFound)


# Next step: for each descriptor start at len(albumDescriptors) and decrement for each additional descriptor so that some descriptors are higher than others for each album
#getDescriptorVectors(albumsDataframe)


# Albums not found (0-999 there are 177)
# [['Lift Yr. Skinny Fists Like Antennas to Heaven!', 'Godspeed You Black Emperor!'], ['The Complete Peel Sessions 1978-2004', 'The Fall'], ['The Bootleg Series Vol. 4: Live 1966 - The "Royal Albert Hall" Concert', 'Bob Dylan'], ['The Great Deceiver: Live 1973-1974', 'King Crimson'], ['Live at Massey Hall 1971', 'Neil Young'], ['Either / Or', 'Elliott Smith'], ['★ [Blackstar]', 'David Bowie'], ['The Pet Sounds Sessions', 'The Beach Boys'], ['Rust Never Sleeps', 'Neil Young & Crazy HorseNeil YoungCrazy Horse'], ['Sunshine Daydream: Veneta, Oregon, August 27, 1972', 'Grateful Dead'], ['On the Beach', 'Neil Young'], ['Velocity : Design : Comfort', 'Sweet Trip'], ['Live Rust', 'Neil Young & Crazy HorseNeil YoungCrazy Horse'], ['Everybody Knows This Is Nowhere', 'Neil Young with Crazy HorseNeil YoungCrazy Horse'], ['Unleashed in the East: Live in Japan', 'Judas Priest'], ['Archive 1967-75', 'Genesis'], ['Ys', 'Joanna Newsom'], ['Live / 1975-85', 'Bruce Springsteen & The E Street BandBruce SpringsteenE Street Band'], ['Sun Bear Concerts Piano Solo: Recorded in Japan', 'Keith Jarrett'], ['The Early Years 1965-1972', 'Pink Floyd'], ['Live at the Fillmore East', 'Neil Young & Crazy HorseNeil YoungCrazy Horse'], ['The Nightwatch: Live at the Amsterdam Concertgebouw November 23rd 1973', 'King Crimson'], ['Sign "☮︎" the Times', 'Prince'], ['Rétrospective Vol. 1 & 2', 'Magma'], ['Dream Letter: Live in London 1968', 'Tim Buckley'], ["Dick's Picks Volume Eight: Harpur College, Binghamton, NY, May 2, 1970", 'Grateful Dead'], ['At the Mountains of Madness', 'Electric Masada'], ["Art Blakey and The Jazz Messengers [Moanin']", 'Art Blakey and The Jazz Messengers'], ['Die Mensch-Maschine', 'Kraftwerk'], ['Coltrane "Live" at the Village Vanguard', 'John Coltrane'], ["Dick's Picks Volume Four: Fillmore East, 2/13-14/70", 'Grateful Dead'], ["'77 Live", 'Les Rallizes Dénudés'], ['MTV Unplugged', 'Alice in Chains'], ['The Noise ザ・ノイズ', 'Hijokaidan'], ['Jimi Plays Monterey: Original Motion Picture Sound Track', 'Jimi Hendrix'], ['Trans Europa Express', 'Kraftwerk'], ['Cowboy Bebop CD-Box', 'Yoko Kanno / Seatbelts菅野よう子 [Yoko Kanno]Seatbelts'], ['Boris at Last -Feedbacker-', 'Boris'], ["C'era una volta il West", 'Ennio Morricone'], ["Donkey Kong Country 2: Diddy's Kong Quest", 'David Wise'], ["ゼルダラの仮面 (The Legend of Zelda: Majora's Mask)", '近藤浩治 [Koji Kondo]'], ['Mother 2: ギーグの逆襲', '田中宏和 [Hirokazu Tanaka] & 鈴木慶一 [Keiichi Suzuki]'], ['Weld', 'Neil Young & Crazy HorseNeil YoungCrazy Hme to Sky Valley]', 'Kyuss'], ['NieR:Automata', '岡部啓一 [Keiichi Okabe] & 帆足圭吾 [Keigo Hoashi]'], ['The Bootleg Series Vol. 6: Live 1964 - Concert at Philharmonic Hall', 'Bob Dylan'], ['Ascension [Edition In Coltrane'], ['Tristan und Isolde (Bayreuth 1966)', 'Chor und Orchester der Bayreuther Festspiele / Karl Böhm / Birgit Nilsson / Wolfgang Windgassen / Christa Ludwig / Martti Talvela / Eberhard WaechterOrchester der Bayreuther FestspieleChor der Bayreuther FestspieleKarl BöhmBirgit NilssonWolfgang WindgassenChrista LudwigMartti TalvelaEberhard Waechter'], ['Black Sabbath Vol. 4', 'Black Sabbath'], ['Live Shit: Binge & Purge', 'Metallica'], ['ファイナルファンタジーVI (Final Fantasy VI)', '植松伸夫 [Nobuo Uematsu]'], ["Tonight's the Night", 'Neil Young'], ['The Nocturnes', 'Artur Rubinstein'], ['塊魂サウンドトラック「塊フォルテack: Katamari Fortissimo Damacy)', '三宅優 [Yuu Miyake]'], ['Flood', 'Boris'], ['Have One on Me', 'Joanna Newsom'], ['Fôrça bruta', 'Jorge Ben'], ['ニンテンドーDS ポケモン ダイヤモンド&パール スーパーミュージックamond & Pearl Super Music Collection)', '増田順一 [Junichi Masuda], 一之瀬剛 [Go Ichinose] & 佐藤仁美 [Hitomi Sato]'], ['The Bootleg Series Vol. 11: The Basement Tapes - Raw', 'Bob Dylan and The Band'], ['PersonShoji Meguro]'], ['Jenny Death: The Powers That B Disc 2', 'Death Grips'], ["C'era una volta in America", 'Ennio Morricone'], ['20 Years of Jethro Tull', 'Jethro Tull'], ['12 Hits From Hell: The MSP Sessions', 'Misfits'], ['John Lennon / Plastic Ono Band', 'John Lennon / Plastic Ono Band'], ['Historic Performances Recorded at the Monterey International Pop Festival', 'Otis Redding / The Jimi Hendrix Experience'], ["Roxy: Tonight's the Night Live", 'Neil Young'], ['もののけ姫 (Mononoke-hime)', '久石譲 [Joe Hisaishi]'], ['London 1986', 'Talk Talk'], ['First Daze Here: The Vintage Collection', 'Pentagram'], ['...And the Ambulancn His Arms', 'Coil'], ['Bowie at the Beeb: The Best of the BBC Radio Sessions 68-72', 'David Bowie'], ['Silent Shout: An Audio Visual Experience', 'The Knife'], ['NieR Gestalt & RepliCant', '岡部啓一 [Keiichi Ok, 石濱翔 [Kakeru Ishihama], 帆足圭吾 [Keigo Hoashi] & 隆文西村 [Takafumi Nishimura]'], ["You're Living All Over Me", 'Dinosaur'], ['嵐 (Arashi)', '山下洋輔トリオ、 大駱駝艦、 ジェラルド大下Yosuke Yamashita Trio大Gerald Oshita'], ['Different Stages / Live', 'Rush'], ['Is There Anybody Out There? The Wall Live 1980-81', 'Pink Floyd'], ['The Bootleg Series Vol. 8: Tell Tale Signs - Rare and Unreleased 1989-2006', 'Bob Dylan'], ['Operation:LIVEcrime', 'Queensrÿche'], ['Rock Dream', 'Boris With Merzbow'], ['Live 1981-82', 'The Birthday Party'], ['Nina Simone in Concert', 'Nina Simone'], ['Live in Copenhagen', 'Mt. Eerie'], ['Wrong', 'NoMeansNo'], ['Song for My Father (Cantiga para meu pai)', 'The Horace Silver Quintet'], ['Visions of the Country', 'Robbie Basho'], ['Recorded Live at the Monterey Jazz Festival', 'John Handy'], ['ワンダと巨咆哮 (Wander and the Colossus: Roar of the Earth)', '大谷幸 [Ko Otani]'], ["Don't Break the Oath", 'Mercyful Fate'], ['Deliquescence', 'Swans'], ["Lamentations: Live at Shepherd's Bush Empire 2003", 'Opeth'], ['(Pneuma)', '青葉市子+三宅純+山本達久+渡辺等青葉市子 [Ichiko Aoba]Jun Miyake山本達久 [Tatsuhisa Yamamoto]Hitoshi Watanabe'], ['Red House Painters [Rollercoaster]', 'Red House Painters'], ['Coma Divine: Recorded L 'Porcupine Tree'], ['Something Else by The Kinks', 'The Kinks'], ['DK Jamz: The Original Donkey Kong Country Soundtrack', 'David Wise / Eveline Fischer'], ['Zuma', 'Neil Young with Crazy HorseNeil YoungCrazy Horse'], ['Bergtatt: Et eeventyr i 5 capitler', 'Ulver'], ['Solo-Concerts: Bremen and Lausanne', 'Keith Jarrett'], ['千と千尋の神隠し (Sen to Chihiro no kamikakushi)', '久石譲 [Joe Hisaishi]'], ['Albert Ayler in Gllage', 'Albert Ayler'], ['Deep Purple in Concert', 'Deep Purple'], ['The 6 String Quartets', 'Takács Quartet'], ["Live at Wembley '86", 'Queen'], ['The Bootleg Series Vol. 7: No Direction Home: The Soundtrack', 'Bob Dylan'], ['Persona4', '目黒将司 [Shoji Meguro]'], ['救済の技法 (Kyuusai no gihou)', '平沢進 [Susumu Hirasawa]'], ['Elvis: TV Special', 'Elvis Presley'], ['Like an Ever Flowing Stream', 'Dismember'], ['Live in Sevilla 2000', 'Masada'], ["Fare Forward Voyagers (Soldier's Choice)", 'John Fahey'], ['Stravinsky Conducts Le Sacre du printemps', 'Columbia Symphony Orchestra / Igor Stravinsky'], ["It's Your World", 'Gil Scott-Heron and Brian Jackson'], ['Vision Creation Newsun', 'Boredoms'], ['Live in Germany 1976', 'Rainbow'], ['Microphones in 2020', 'The Microphones'], ['The Lord of the Rings: The Return of the King', 'Howard Shore'], ['Way Down in the Rust Bucket', 'Neil Young with Crazy HorseNeil YoungCrazy Horse'], ['Dead as Dreams', 'Weakling'], ['Disco Elysium', 'British Sea Power'], ['Masked Dancers: Concern in So Many Things You Forget Where You Are', 'The Brave Little Abacus'], ['Symphonie No. 9 / Moldau', 'Wiener Philharmoniker / Herbert von Karajan'], ['Twin Peaks: Limited Event Series Soundtrack', 'Angelo Badalamenti'], ['"Windswept Adan” Concert at Bunkamura Orchard Hall', '青葉市子 [Ichiko Aoba]'], ['Tango: Zero Hour / Nuevo Tango: Hora Zero', 'Astor Piazzolla and The New Tango Quintet'], ['Unquestionable Presence', 'Atheist'], ['Live at the Old Waldorf, San Francisco, 6/29/78', 'Television'], ['Die Kunst der Fuge', 'Tatiana Nikolayeva'], ['Live Kreation', 'Kreator'], ['Persona3', '目黒将司 [Shoji Meguro]'], ['"Evergrace"', '星野康太 [Kota '], ['Os afro-sambas de Baden e Vinícius', 'Baden Powell & Vinicius de Moraes'], ['Sechs Suiten für Violoncello Solo BWV 1007-1012', 'Pierre Fournier'], ['Town Hall Concert 1964, Vol. 1', 'Charles Mingus'], ['Music for 18 Musicians', 'Grand Valley State University New Music Ensemble'], ['4 Balladen; Barcarolle; Fantasie', 'Krystian Zimerman'], ['剣風伝奇ベルセルク (Berserk)', '平沢進 [Susumu Hirasawa]'], ['The Dance ofd the Sun', 'Natural Snow Buildings'], ['Kingdom Hearts II Original Soundtrack', '下村陽子 [Yoko Shimomura]'], ['The Legend of Zelda: Breath of the Wild', '片岡真央 [Manaka Kataoka] / 岩田恭明 [Yasuaki Iwata] / Wakai]'], ['Live at the Cellar Door', 'Neil Young'], ['R4™ / Ridge Racer Type 4 / Direct Audio', '*Namco Consumer Software Sound Team'], ['Mulholland Dr.', 'Angelo Badalamenti & David Lynch'], ['Court and Spark', 'Joni Mitchell'], ['My Funny Valentine: Miles Davis in Concert', 'Miles Davis'], ['ABC Music: The Radio 1 Sessions', 'Stereolab'], ['무너지기 (Crumbling)', '공중도둑 [Mid-Air Thief]'], ['Feldman Edition 6: Strtet No. 2', 'FLUX Quartet'], ['悪魔城ドラキュラX～月下の夜想曲～ (Akumajo Dracula X ~Gekka no Nocturne~)', '山根ミチル [Michiru Yamane]'], ['アダンの風 (Windswept Adan)', '青葉市子 [Ichiko Aoba]'], ['First Meditoltrane'], ["Apostrophe (')", 'Frank Zappa'], ["'Four' & More: Recorded Live in Concert", 'Miles Davis'], ['Mi sei apparso come un fantasma', 'Songs: Ohia'], ['渋星 (Shibuboshi)', '渋さ知らズ [Shibusashirazu]'],ルファンタジーX (Final Fantasy X)', '仲野順也 [Junya Nakano] / 浜渦正志 [Masashi Hamauzu] / 植松伸夫 [Nobuo Uematsu]'], ['In Person at the Whisky a Go Go', 'Otis Redding'], ['Remote Utopias: 2nd May 2020', 'Blad Ecco2K'], ['Symphony No. 3 (Symphony of Sorrowful Songs); 3 Olden Style Pieces', 'Polish National Radio Symphony Orchestra / Antoni Wit / Zofia KilanowiczNarodowa Orkiestra Symfoniczna Polskiego Radia w KatowicachAntoni WitZofia Kilanowicz'], ['Sonic the Hedgehog CD Original Soundtrack 20th Anniversary Edition', '尾形雅史 [Masafumi Ogata] / 幡谷尚史 [Naofumi Hataya]'], ['Live at Benaroya Hall', 'Pearl Jam'], ['Fa-tal:odo vapor', 'Gal Costa'], ['ピンポン オリジナルサウンドトラック', '牛尾憲輔 [Kensuke Ushio]'], ['Root Down: Jimmy Smith Live!', 'Jimmy Smith'], ["If You're Feeling Sinister: Live at The Barbican", 'Belle & Sebasoanna Newsom'], ['Complete / Sämtliche / Les 21 Nocturnes', 'Claudio Arrau']]

# Albums not found (1000-1999 there are 180)
# [['Xenogears', '光田康典 [Yasunori Mitsuda]'], ['ゼノブレイド Xenoblade', '下村陽子 [Yoko Shimomura] / 清田愛未 [Manami Kiyota] / ACE+'], ['3 Gymnopédies & Other Piano Works · und andere Klavierstücke', 'Pascal Champloo Music Record: Departure', 'Nujabes / Fat Jon'], ['Haibane-Renmei: Hanenone', '大谷幸 [Ko Otani]'], ['Keeper of the Seven Keys Part II', 'Helloween'], ["Can't Buy a Thrill", 'Steely Dan'], ["Walter CarloClockwork Orange", 'Wendy Carlos'], ['Voyager 1', 'Verve'], ['Jet Set Radio Future (ジェットセットラジオフューチャー)', '長沼英樹 [Hideki Naganuma]'], ['Choir Concerto', 'Russian State Symphonic Cappella / Valerвенный камерный хор Министерства культуры СССР [USSR Ministry of Culture Chamber Choir]Валерий Полянский [Valeri Polyansky]'], ['E·MO·TION', 'Carly Rae Jepsen'], ['フリクリ (FLCL): OST 1 ~ Addict', 'the pillows''Die Kunst der Fuge', 'Evgeni Koroliov'], ['The Wicker Man', 'Paul Giovanni / Gary Carpenter / Magnet'], ['Dark to Themselves', 'Cecil Taylor Unit'], ['真・女神転生III Nocturne (Shin Megami Tensei III: Nocturne)将司 [Shoji Meguro] / 田崎寿子 [Toshiko Tasaki] / 土屋憲一 [Kenichi Tsuchiya]'], ['Ascension [Edition II]', 'John Coltrane'], ['Neue Wiener Schule: Die Streichquartette', 'LaSalle Quartett'], ['Lola Versus Powere Moneygoround (Part One)', 'The Kinks'], ["I'm Still in Love With You", 'Al Green'], ['Metal Box', 'PiL'], ['Peace and Love', 'Dadawah'], ['Tropicália ou panis et circencis', 'Various Artists'], ['Jacques Brel [Ces gens-là]', 'Jacques Brel'], ["Tim Burton's The Nightmare Before Christmas", 'Danny Elfman'], ['Arnold Schoenberg 2: Streichquartette I-IV', 'Arditti String Quartet / Dawn Upshaw'], ['The Concert for Bangla Desh', 'Various Artists'], ['Dopesmoker', 'Sleep'], ['Symphonie Nr. 6 »Pastorale«', 'Wiener Philharmoniker / Karl Böhm'], ['Симфония № 5', 'Академический симфонический оркестр Московской государственной филармонии / Кирилл Кондрашин [Kirill Kondrashin]'], ['Requiem; Lontano; Continuum', 'Sinfonieorchester des Hessischen Rundfunks / Chor des Bayerischen Rundfunks / Sinfonieorchester des Südwestfunks / Michael Gielen / Wolfgang Schubert / Ernest Bour / Liliana Poli / Barbra Ericson / Antoinette Vischerhr-SinfonieorchesterChor des Bayerischen RundfunksSWR SinfonieorchesterMichael GielenWolfgang SchubertErnest BourLiliana PoliBarbro EricsonAntoinette Vischer'], ['50th Birthday Celebration Vol. 4', 'Electric Masada'], ['Young Shakespeare', 'Neil Young'], ["Didn't It Rain", 'Songs: Ohia'], ['Performing "flood"', 'Boris'], ['The Jazz Messengers at the Cafe Bohemia, Volume 1', 'The Jazz Messengers'], ['Alchemy: Dire Straits Live', 'Dire Straits'], ["Long Season '96~7 96.12.26 Akasaka Blitz", 'Fishmans'], ['あらためまして、はじめまして、ミドリです。ashite, hajimemashite, Midori desu.)', 'ミドリ [Midori]'], ['Biomech', 'Ocean Machine'], ['Fas – Ite, maledicti, in ignem aeternum', 'Deathspell Omega'], ['Le mystère des voix bulgares : volume 1', 'Le Mystère dvoix bulgares'], ['Deaf Dumb Blind (Summun Bukmun Umyun)', 'Pharoah Sanders'], ['At the "Golden Circle" Stockholm, Volume One', 'The Ornette Coleman Trio'], ['Street Fighter III: 3rd Strike - Fight for the Future', '奥河英樹 [Hideki Okugawa]'], ['The Fragile: Deviations 1', 'Nine Inch Nails'], ['化物語 音楽全集 Songs & Soundtracks (Bakemonogatari Ongaku Zenshuu Songs & Soundtracks)', '神前暁 [Satoru Kosaki]'], ['Get innry Rollins'], ['6 String Quartets', 'Emerson String Quartet'], ["Spirit They're Gone Spirit They've Vanished", 'Avey Tare and Panda BearAvey TarePanda Bear'], ['Complete Works Opp 1-31', 'Pierre Boulez'], ["Sonny's Dream (Birth of the New Cool)", 'The Sonny Criss Orchestra'], ['R.E.M. Live at the Olympia', 'R.E.M.'], ['Live Four', 'Coil'], ['Friday Night in San Francisco', 'Al Di Meola, John McLaughlin & Paco de Lucía'], ['Buhloone Mind State', 'De La Soul'], ['Revés / Yosoy', 'Café Tacvba'], ['Jet Set Radio', '長沼英樹 [Hideki Naganuma]'], ['Live: Bursting Out', 'Jethro Tull'], ['Live! In Tune and on Time', 'DJ Shadow'], ['ドーDS ポケモンブラック・ホワイト スーパーミュージックコレクション (Nintendo DS Pokémon Black・White Super Music Collection)', '景山将太 [Shota Kageyama]'], ["We're Only in It for the Money", 'The Mothers of Inv ['Les cinq quatuors à cordes; Trio à cordes; Khoom', 'Quatuor Arditti / Aldo Brizzi / Michiko Hirayama / Maurizio Ben Omar / Frank LloydArditti String QuartetAldo BrizziMichiko HirayamaMaurizio Ben OmarFrank Lloyd'], ['Athenaeum, Homebush, Quay & Raab', 'The Necks'], ['革命京劇 (Revolutionary Pekinese Opera) Ver.1.28', 'Ground-Zero'], ['Horse Rotorvator', 'Coil'], ['The Hissing of Summer Lawns', 'Joni Mitchell'], ['かKagayaki)', '高木正勝 [Masakatsu Takagi]'], ['Klavierkonzert Nr. 3 C-dur / Klavierkonzert G-dur', 'Berliner Philharmoniker / Claudio Abbado / Martha Argerich'], ['Live Wire / Blues Power', 'Albert King'], ['ゼル風のタクト～ (The Legend of Zelda: The Wind Waker)', '永田権太 [Kenta Nagata], 若井淑 [Hajime Wakai], 峰岸透 [Toru Minegishi] & 近藤浩治 [Koji Kondo]'], ["If I Could Do It All Over Again, I'd Do It All Over You"mika', 'Djeli Moussa Diawara'], ['The Well-Tuned Piano 81 X 25, 6:17:50 - 11:18:59 PM NYC', 'La Monte Young'], ['Down on the Road by the Beach', 'Steve Hiett'], ['Germfree Adolescents', 'X-Ray Spex'], ['24 Préludes, Op. 28; Préludes Nr. 25, Op. 45; Nr. 26, Op. Posth.', 'Martha Argerich'], ['Requiem KV 626', 'Berliner Philharmoniker / Wiener Singverein / Herbert von Karajan / Wilma Lipp / Hilde Rössl-Majdan / Anton Dermota / Walter BerryBerliner PhilharmonikerWiener SingvereinHerbert von KarajanWilma LippHilde Rössel-MajdanAnton DermotaWalter Berry'], ['BBC Radio Theatre, London, June 27, 2000', 'David Bowie'], ["It'll All Work Out in Boomland", 'T2'], ['Radio One', 'Jimi Hendrix Experience'], ['Theatre Royal Drury Lane 8th September 1974', 'Robert Wyatt & Friends'], ['Tashi Plays Messiaen: Quartet for the End of Time', 'Tashi'], ['Les stances à Sophie', 'Art Ensemble of Chicago'], ['Stakes Is High', 'De La Soul'], ['Novos Baianos F.C.', 'Novos Baianos'], ['天使のたまご 音楽編 / 水に棲む (Tenshi no tamago ongaku hen / Mizu ni sumu)', '菅野由nno]'], ['Autechre [LP5]', 'Autechre'], ["Burritos, Inspiration Point, Fork Balloon Sports, Cards in the Spokes, Automatic Biographies, Kites, Kung Fu, Trophies, Banana Peels We've Slipped On and Egg Shells We've Tippy Toed Over", "Cap'n Jazz"], ['Голос сталі', 'Nokturnal Mortum'], ['Deus Ex: Game of the Year Edition', 'Alexander Brandon / Michiel van den Bos'], ['Milagre dos peixes', 'Milton Nascimento'], ["Vingt regards sur l'enfant-Jésus", 'Steven Osborne'], ['Black Antlers', 'Coil'], ['Live at Primavera Sound 2012', 'James Ferraro'], ['あくまのうた (Akuma no uta)', 'Boris'], ['Conference of the Birds', 'David Holland Quar ['György Ligeti Edition 3: Works for Piano - Études, Musica ricercata', 'Pierre-Laurent Aimard'], ['24 Préludes, Op. 28', 'Maurizio Pollini'], ['Queen on Fire: Live at the Bowl', 'Queen'], ['映画 聲の形 オリジナック A Shape of Light', '牛尾憲輔 [Kensuke Ushio]'], ['3 Feet High and Rising', 'De La Soul'], ['The Nocturnal Silence', 'Necrophobic'], ['のぶえの海 (Nobue no umi)', '河名伸江 [Nobue Kawana]'], ['Miles Davis in Night at the Blackhawk, San Francisco, Volume 1', 'Miles Davis'], ['Below the Heavens', 'Blu & Exile'], ['Bootleg Series Volume 1: The Quine Tapes', 'The Velvet Underground'], ['放課後ティータイムII (Ho-kago Tecond)', '放課後ティータイム [Hokago Tea Time]'], ['Quattro pezzi per orchestra; Anahit; Uaxuctum', 'Orchestre et Chœur de la Radio-Télévision Polonaise de Cracovie / Jürg Wyttenbach / Carmen Fournier / Tristan MurailOrkiestra Polskiego Radia w KrakowieChór Polskiego Radia w KrakowieJürg WyttenbachCarmen FournierTristan Murail'], ['Live in L.A. (Death & Raw)', 'Death'], ['Keeper of the Seven Keys Part I', 'Helloween'], ['Nina Simone at Town Hall', 'Nina Simone'], ['Geraes', 'Milton Nascimento'], ['Last Date', 'Eric Dolphy'], ['東方紅魔郷 ～ The Embodiment of Scarlet Devil. (Touhou Koumakyou)', '上海アリス幻樂団'], ["Suite bergren's Corner; Estampes; L'isle joyeuse; La fille aux cheveux de lin; La plus que lente; Etude pour les arpèges", 'Alexis Weissenberg'], ["Œuvres d'Erik Satie", 'Aldo Ciccolini / Gabriel Tacchino'], ['Nothingface', 'Voivod'], ['Poppy Nogood and the Phantom Band: All Night Flight Vol.1', 'Terry Riley'], ['Live at the Troubadour 1969', 'Tim Buckley'], ['Concerto grosso I; Concerto for Oboe and Harp; Concerto for Piano and Strings', 'New Stockholm Chamber Orchestra / Lev MarkizStockholms nya kammarorkesterLev Markiz'], ['The Milk-Eyed Mender', 'Joanna Newsom'], ['Paris, Texas', 'Ry Cooder'], ['Blast From the Past', 'Gamma Ray'], ['20 palavras ao redor do Sol', 'Cátia de França'], ['Hardcore Devo, Vol. 1: 74-77', 'Devo'], ['Warszawa', 'Porcupine Tree'], ['BBC Sessions', 'Cocteau Twins'], ['Chewed Up', 'Louis C.K.'], ['Meat Puppets II', 'Meat Puppets'], ['Ewa Demarczyk śpiewa piosenki Zygmunta Koniecznego', 'Ewa Demarczyk'], ['Niggas on the Moon: The Powers That B Disc 1', 'Death Grips'], ['Never Turn Your Back on a Friend', 'Budgie'], ['Tindersticks [II]', 'Tindersticks'], ['The Godfather, Part II', 'Nino Rota / Carmine Coppola'], ['I See a Darkness', "Bonnie 'Prince' Billy"], ['Kraan Live', 'Kraan'], ['Parental Advisory: Explicit Lyrics', 'George Carlin'], ['Naked City', 'John Zorn'], ['Different Trains; Electric Counterpoint', 'Kronos Quartet / Pat Metheny'], ['Adnos I-III', 'Eliane Radigue'], ["Isn't Anything", 'My Bloody Valentine'], ['For Philip Guston', 'The California EAR Unit'], ['The Incredible Jazz Guitar of Wes Montgomery', 'Wes Montgomery'], ["Da Devil's Playground: Underground Solo", 'Koopsta Knicca'], ['At the "Golden Circle" Stockholm, Volume Two', 'The Ornette Coleman Trio'], ['4 Way Street', 'Crosby, Stills, Nash & Young'], ["I'm in Your Mind Fuzz", 'King Gizzard & the Lizard Wizard'], ['Darwin!', 'Banco del Mutuo Soccorso'], ['Winterkaelte', "Paysage d'Hiver"], ['Rothko Chapel; Why Patterns?', 'UC Berkeley Chamber Chorus / The California EAR Unit / Philip Brett / David Abel / Karen Rosenak / William Winant / Dorothy Stone / Arthur Jarvinen / Gaylord Mowrey'], ['Volume 6: Days Have Gone By', 'John Fahey'], ['HBO (Haitian Body Odor)', 'Mach-Hommy'], ['Ladies of the Canyon', 'Joni Mitchell'], ['The Gate', 'Swans'], ['Préludes · Volume 1', 'Arturo Benedetti Michelangeli'], ['Miles of Aisles', 'Joni Mitchell and the L.A. ExpressJoni MitchellL.A. Express'], ['Chopin / Brahms / Liszt / Ravel / Prokofieff', 'Martha Argerich'], ['Il était une forêt...', 'Gris'], ['大勘定 (Daikanjyo)'明田川荘之 [Shoji Aketagawa], 三上寛 [Kan Mikami] & 石塚俊明 [Toshiaki Ishizuka]'], ['Vespro della Beata Vergine 1610', 'La Capella Reial / Coro del Centro di Musica Antica di Padova / Jordi Savall / Montserrat uy de Mey / Maria Cristina Kiehr / Gerd Türk / Gian Paolo Fagotto / Paolo Costa / Pietro Spagnoli / Daniele CarnovichLa Capella Reial de CatalunyaCoro del Centro di Musica Antica di PadovaJordi SavallMontserrat FiguerasGuy de MeyMaria Cristina KiehrGerd TürkGian Paolo FagottoPaolo CostaPietro SpagnoliDaniele Carnovich'], ['Alturas de Machu Pichu', 'Los Jaivas'], ['Caesar Demos', 'Ween'], ['GBAポケモン ルビー&サファイアコンプリート (GBA Pokémon Ruby & Sapphire Music Super Complete)', '増田順一 [Junichi Masuda], 一之瀬剛 [Go Ichinose] & 青木森一 [Morikazu Aoki]'], ['ハウルの動く城 (Hauru no ugoku shiro)', '久石譲 [Joe Hisaishi]rgeGilberto GilJorge Ben'], ['Time Fades Away', 'Neil Young'], ["The Devil's Playground", 'Da Koopsta Knicca'], ['Coltrane Live at the Village Vanguard Again!', 'John Coltrane'], ['Purple Sun', 'Tomasz Stanko Quintet'], ['Poland: The Warsaw Concert', 'Tangerine Dream'], ['Crippled Symmetry', 'Dietmar Wiesner / Markus Hinterhäuser / Robyn Schulkowsky'], ['平成風俗 (Heisei fūzoku)', '椎名林檎 × 斎藤ネコ椎名林檎 [Sheena RSaito]'], ['In concerto: Arrangiamenti PFM', 'Fabrizio De André con PFMFabrizio De AndréPremiata Forneria Marconi'], ['Quintett C-Dur', 'Alban Berg Quartett / Heinrich Schiff'], ['Midtown 120 Blues', 'DJ Sprinkles'], ['The Ghost~Pop Tape', 'Devon Hendryx'], ['Choirs of the Eye', 'Kayo Dot']]




