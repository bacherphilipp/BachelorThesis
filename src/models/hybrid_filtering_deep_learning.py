import ast
from sklearn.model_selection import train_test_split
import pandas as pd
import re
import numpy as np
from keras.layers import *
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import configparser
from src.models.helpers.user_genre_profile import UserGenreProfileGenerator
from src.data_retrieval.dbconnect import get_connection

def get_music_data():
    """Retrieves music data containing track features. Can be limites via the data_record_limit setting

    Parameters:

    Returns:

    """
    print('Fetching music data...')
    cursor = get_connection()
    config = configparser.ConfigParser()
    config.read('config/settings.ini')
    data_record_limit = config.getint('general', 'data_record_limit', fallback=0)
    limit = '' if data_record_limit == 0 else ' limit ' + str(data_record_limit)

    cursor.execute('''
        select
            tracks.track_id, genres
        from
            tracks
            join genreannotation on tracks.track_id = genreannotation.track_id
        where  
            tracks.track_id in (select track_id from events)
    ''' + limit)
    print('Music data fetched...')
    return cursor.fetchall()

def prepare_data():
    """Prepares the data to be used by the hybrid filtering model. 

    Parameters:

    Returns:
    
    """
    generator = UserGenreProfileGenerator()
    user_profile = generator.build_profile(False)
    user_profile_matrix = user_profile.pivot(index='user_id', columns='genres', values='normalized_rating').fillna(0).reset_index()

    used_genres = [col for col in user_profile_matrix.columns if col not in ['user_id']]    
    
    raw_music_data = pd.DataFrame(get_music_data(), columns=['track_id', 'genres'])
    raw_music_data['genres'] = raw_music_data['genres'].apply(ast.literal_eval)
    music_data_exploded = raw_music_data.explode('genres')
    music_data_exploded['value']=1
    music_data_exploded = music_data_exploded.drop_duplicates(subset=['track_id', 'genres'])
    
    song_genre_matrix = music_data_exploded.pivot(index='track_id', columns='genres', values='value').fillna(0).reset_index()

    for genre in used_genres:
        if genre not in song_genre_matrix.columns:
            song_genre_matrix[genre] = 0

    final_columns = ['track_id'] + used_genres
    song_genre_matrix = song_genre_matrix[final_columns]

    return user_profile_matrix, song_genre_matrix

def create_metadata_df(df, feature_columns, embedding_columns):
    """Creates a dataframe containing the user/item metadata

    Parameters:
    feature_columns: list of column names that contain single features values
    embedding_columns: list of column names that contain vector embeddings

    Returns:
    metadata DataFrame
    """
    features = df[feature_columns].reset_index(drop=True)
    embeddings = pd.DataFrame()
    for column in embedding_columns:
        embeddings = pd.concat([embeddings, pd.DataFrame(np.vstack(df[column]))], axis=1)
    result = pd.concat([features,embeddings],axis=1)
    return result

def hybrid_recommender_v1(n_item_features, n_user_features, embedding_size, n_users, n_items):
    """Creates the deep learning model used for the hybrid filtering

    Parameters:
    n_item_features: Number of item features
    n_user_features: Number of user features
    embedding_size: Embedding size for the model
    n_users: Number of users in the dataset
    n_items: Number of items in the dataset

    Returns:
    The compiled model
    """
    user_id_input = Input(shape=[1], name='user')
    item_id_input = Input(shape=[1], name='item')
    item_meta_input = Input(shape=[n_item_features], name='item_features')
    user_meta_input = Input(shape=[n_user_features], name='user_features')

    user_embedding = Embedding(output_dim=embedding_size, input_dim=n_users, name='user_embedding')(user_id_input)
    item_embedding = Embedding(output_dim=embedding_size, input_dim=n_items, name='item_embedding')(item_id_input)
    item_metadata = Dense(units=embedding_size, name='item_metadata')(item_meta_input)
    user_metadata = Dense(units=embedding_size, name='user_metadata')(user_meta_input)

    user_vec = Flatten()(user_embedding)
    item_vec = Flatten()(item_embedding)
    item_vec = Add()([item_vec, item_metadata])
    user_vec = Add()([user_vec, user_metadata])

    input_vec = Concatenate()([user_vec, item_vec])#, item_metadata, user_metadata])

    x = Dense(128, activation='relu')(input_vec)
    x = Dropout(0.5)(x)
    y = Dense(1)(x)

    model = Model(inputs=[user_id_input, item_id_input, item_meta_input, user_meta_input], outputs=y)
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001), metrics=['mae'])
    return model

def evaluate_byms_groups(model, test_data, genres_user, genres_track):
    """Evaluates the results of the DL model by usergroup. Calculates RMSE and MAE

    Parameters:
    model (Model): the trained Model
    test_data (DataFrame): the test set portion of the useritem relation set

    Returns:
    
    """

    groups = dict()
    groups['all'] = [-1]
    groups['ms'] = test_data[test_data["isbyms"] == 0]#[['user_id', 'track_id', 'rating']]
    groups['byms'] = test_data[test_data["isbyms"] == 1]#[['user_id', 'track_id', 'rating']]
    groups['UGR0'] = test_data[test_data["usergroup"] == 0]#[['user_id', 'track_id', 'rating']]
    groups['UGR1'] = test_data[test_data["usergroup"] == 1]#[['user_id', 'track_id', 'rating']]
    groups['UGR2'] = test_data[test_data["usergroup"] == 2]#[['user_id', 'track_id', 'rating']]
    groups['UGR3'] = test_data[test_data["usergroup"] == 3]#[['user_id', 'track_id', 'rating']]

    for group_name, data in groups.items():
        if group_name != 'all':  
            if not data.empty:
                user_ids = np.array([pair[0] for pair in data.values.tolist()])
                track_ids = np.array([pair[1] for pair in data.values.tolist()])

                item_feature_cols = genres_track # item feature columns that contain a single value
                item_embedding_cols = [] # item feature columns that contain a list of embeddings - applicable to image or text embeddings
                user_feature_cols = genres_user # gender plus additional one-hot-encoded features
                user_embedding_cols = []

                test_item_metadata = create_metadata_df(data, item_feature_cols, item_embedding_cols)
                test_user_metadata = create_metadata_df(data, user_feature_cols, user_embedding_cols)

                user_ids = user_ids.reshape(-1, 1) 
                track_ids = track_ids.reshape(-1, 1) 

                predictions = model.predict([user_ids, track_ids, test_item_metadata, test_user_metadata])

                predictions = [item for sublist in predictions.tolist() for item in sublist]
   
                true_ratings = [pair[2] for pair in data.values.tolist()]

                pred_ratings =  [pred for pred in predictions]

                print(f'RMSE for {group_name}:')
                print( np.sqrt(np.mean((np.array(true_ratings)-np.array(pred_ratings))**2)))

                print(f'MAE for {group_name}:')
                print(np.mean(np.abs(np.array(true_ratings) - np.array(pred_ratings))))

            else:
                print(f'No data to predict for group {group_name}')  

def prepare_model(ratings, users, items):
    """Prepares the training data for the model and fits it

    Parameters:
    ratings: The user-item interaction rating DataFrame
    users: The user DataFrame including the user profile
    items: The item DataFrame including tracks and genre information

    Returns:
    
    """
    users.rename(columns={col: f'user_{col}' for col in users.columns if col not in ['genres', 'user_id']}, inplace=True)
    ratings = ratings[ratings['track_id'].isin(items['track_id'])]
    ratings = ratings[ratings['user_id'].isin(users['user_id'])]
    items = items[items['track_id'].isin(ratings['track_id'])]
    ratings = pd.merge(ratings, items, on='track_id')
    ratings = pd.merge(ratings, users, on='user_id')

    user_encoder = LabelEncoder()
    track_encoder = LabelEncoder()

    items['track_id'] = track_encoder.fit_transform(items['track_id'])
    ratings['track_id'] = track_encoder.transform(ratings['track_id'])

    users['user_id'] = user_encoder.fit_transform(users['user_id'])
    ratings['user_id'] = user_encoder.transform(ratings['user_id'])

    train, test = train_test_split(ratings, test_size=0.2, random_state=42)
    n_users = len(ratings.user_id.unique())
    n_items = len(ratings.track_id.unique())
    cols_except_genres_userid = [col for col in users.columns if col not in ['genres', 'user_id']]
    cols_except_genres_trackid = [col for col in items.columns if col not in ['genres', 'track_id']]
    item_feature_cols = cols_except_genres_trackid 
    item_embedding_cols = []
    user_feature_cols = cols_except_genres_userid 
    user_embedding_cols = []

    train_item_metadata = create_metadata_df(train, item_feature_cols, item_embedding_cols)
    test_item_metadata = create_metadata_df(test, item_feature_cols, item_embedding_cols)

    train_user_metadata = create_metadata_df(train, user_feature_cols, user_embedding_cols)
    test_user_metadata = create_metadata_df(test, user_feature_cols, user_embedding_cols)

    n_item_features = len(cols_except_genres_trackid)
    n_user_features = len(cols_except_genres_userid)
    embedding_size = 256

    model = hybrid_recommender_v1(n_item_features, n_user_features, embedding_size, n_users, n_items)

    print("Training start")

    model.fit([train.user_id, train.track_id, train_item_metadata, train_user_metadata]
                        , train.rating
                        , batch_size=128, epochs=30
                        , validation_split=0.2
                        , validation_data=([test.user_id, test.track_id, test_item_metadata, test_user_metadata], test.rating)
                        , callbacks = []
                        , shuffle=True)

    evaluate_byms_groups(model, test, cols_except_genres_userid, cols_except_genres_trackid)

def base_hybrid_filtering_deep_learning():
    """Entry function for hybrid filtering using deep learning.

    Parameters:

    Returns:

    """
    generator = UserGenreProfileGenerator()
    ratings = generator.get_data(False)
    ratings = pd.DataFrame(ratings, columns=['user_id', 'track_id', 'rating', 'usergroup', 'isbyms', 'genres' ])

    users, items = prepare_data()

    prepare_model(ratings, users, items)
    




