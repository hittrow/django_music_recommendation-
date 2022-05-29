from my_app import Evaluation as Evaluation
from my_app import Recommenders as Recommenders
import joblib
import time
import numpy as np
from sklearn.model_selection import train_test_split
import pandas
##get_ipython().run_line_magic('matplotlib', 'inline')

def get_music_data_test(user_input):
    # # Load music data

    # Read userid-songid-listen_count triplets
    triplets_file = 'C:/Users/Harshit/Desktop/Python/Music Dataset Recommender/10000.txt'
    songs_metadata_file = 'C:/Users/Harshit/Desktop/Python/Music Dataset Recommender/song_data.csv'

    song_df_1 = pandas.read_table(triplets_file, header=None)
    song_df_1.columns = ['user_id', 'song_id', 'listen_count']


    # Read song  metadata
    song_df_2 = pandas.read_csv(songs_metadata_file)


    # Merge the two dataframes above to create input dataframe for recommender systems
    song_df = pandas.merge(song_df_1, song_df_2.drop_duplicates(
        ['song_id']), on="song_id", how="left")


    # # Explore data
    #
    # Music data shows how many times a user listened to a song, as well as the details of the song.

    song_df.head()

    # ## Length of the dataset


    len(song_df)

    # ## Create a subset of the dataset
    song_df = song_df.head(10000)

    # Merge song title and artist_name columns to make a merged column
    song_df['song'] = song_df['title'].map(str) + " - " + song_df['artist_name']


    # ## Showing the most popular songs in the dataset
    song_grouped = song_df.groupby(['song']).agg(
        {'listen_count': 'count'}).reset_index()
    grouped_sum = song_grouped['listen_count'].sum()
    song_grouped['percentage'] = song_grouped['listen_count'].div(grouped_sum)*100
    song_grouped.sort_values(['listen_count', 'song'], ascending=[0, 1])


    # ## Count number of unique users in the dataset
    users = song_df['user_id'].unique()
    len(users)


    # Counting the number of unique songs in the dataset

    # Fill in the code here
    songs = song_df['song'].unique()
    len(songs)


    # # Create a song recommender
    train_data, test_data = train_test_split(
        song_df, test_size=0.20, random_state=0)
    print(train_data.head(5))


    # Recommenders.item_similarity_recommender_py

    # ### Create an instance of item similarity based recommender class
    is_model = Recommenders.item_similarity_recommender_py()
    is_model.create(train_data, 'user_id', 'song')


    # ### Use the personalized model to make some song recommendations

    # Print the songs for the user in training data
    user_id = users[5]
    user_items = is_model.get_user_items(user_id)
    #
    print("------------------------------------------------------------------------------------")
    print("Training data songs for the user userid: %s:" % user_id)
    print("------------------------------------------------------------------------------------")

    for user_item in user_items:
        print(user_item)

    print("----------------------------------------------------------------------")
    print("Recommendation process going on:")
    print("----------------------------------------------------------------------")

    # Recommend songs for the user using personalized model
    is_model.recommend(user_id)
    # Use the personalized recommender model to get similar songs for the following song.

    # Fill in the code here
    data_to_return = is_model.get_similar_items([user_input])
    # import ipdb; ipdb.set_trace()
    return(data_to_return)
