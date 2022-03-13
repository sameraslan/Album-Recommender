# Album-Recommender

This is an album recommender system using the top charted albums of all time on RateYourMusic.

1. Obtained a list of top 5,000 albums and their metadata from RateYourMusic and read them into a CSV. The metadata for each album includes release date, primary genre(s), overall rating, number of ratings, and number of reviews.

2. Obtained the audio metrics for every track on each album using the spotify api spotipy library.

3. Cleaned the data, assigned numerical values to non-numerical features/parameters, normalized un-normalized values.

3. Developed an unsupervised K-Nearest-Neighbours model (using SKlearn). The general idea can be thought of simplest in 2-dimensions. For example, x = Energy (a spotify audio track metric) and y = Danceability (another spotify audio track metric). Each album on the graph is defined as a point, with some Energy x and some Danceability y. In an ideal model, albums on this x,y graph would be clustered by genre (ex. Jazz albums close to one another, Indie-Rock, Soul, etc.). In reality, the model has a non-visualizable number of dimensions, one for each feature. Albums are points in some n-dimensional space.

For example, inputting the album Revoler by The Beatles, the following 5 albums to its right are the most recommended based on the model I developed (most to least similar from left to right).
![Screen Shot 2022-02-20 at 9 43 04 PM](https://user-images.githubusercontent.com/82460915/154880486-335e4139-bd6d-4d45-83e9-dd8aac41df24.jpg)


Or Madvillainy by Madvillain
![Screen Shot 2022-02-20 at 9 38 13 PM](https://user-images.githubusercontent.com/82460915/154880035-b1967fc0-a911-4393-85d9-306f4498a907.jpg)


4. Now in process of playing with features and weights for an ideal and accurate model. 

5. Created an album similarity map using Multidimensional Scaling, a method that allows for visualizing the level of similarity between albums in 2-D. It is an optimization algorithm that minimizes the stress loss function (a residual sum of squares). Intuitively, this can be thought of as minimizing the order of distances between albums in a high-dimensional space and the distance between them in a 2-D space. The image below displays the result of this as a map of albums. Close albums have similar audio metrics and RYM descriptors. To zoom in and view the map in high quality, navigate to /SimilarityMap/SimilarityMap.jpg .
![Screen Shot 2022-03-13 at 5 14 53 PM](https://user-images.githubusercontent.com/82460915/158079540-10cd2c5c-a00b-42ee-bc5c-e4a77d02d5be.jpg)
