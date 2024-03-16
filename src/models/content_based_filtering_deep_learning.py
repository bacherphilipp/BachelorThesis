import ast
import configparser
import itertools

import numpy as np
from src.models.helpers.user_genre_profile import UserGenreProfileGenerator
from tensorflow.keras.layers import Input, Dot
from src.data_retrieval.dbconnect import get_connection
import tensorflow as tf
import pandas

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

def get_interaction_data():
    """Retrieves music data containing track features. Can be limites via the data_record_limit setting

    Parameters:

    Returns:

    """
    print('Fetching interaction data...')
    cursor = get_connection()
    config = configparser.ConfigParser()
    config.read('config/settings.ini')
    data_record_limit = config.getint('general', 'data_record_limit', fallback=0)
    limit = '' if data_record_limit == 0 else ' limit ' + str(data_record_limit)

    cursor.execute('''
        select 
            user_id, track_id, 1 
        from 
            events
        group by
            user_id, track_id
    ''' + limit)
    print('Interaction data fetched...')
    return cursor.fetchall()

def generate_negative_samples(interaction_data, users, songs, ratio=1):
    negative_samples = []
    for user_id in users:
        user_tracks = interaction_data[interaction_data['user_id'] == user_id]['track_id'].values
        
        potential_negatives = np.setdiff1d(songs, user_tracks)
        
        num_negatives = min(len(user_tracks) * ratio, len(potential_negatives))
        negative_track_ids = np.random.choice(potential_negatives, size=num_negatives, replace=False)
        
        for track_id in negative_track_ids:
            negative_samples.append([user_id, track_id, 0])
    
    negative_samples_df = pandas.DataFrame(negative_samples, columns=interaction_data.columns)
    
    return negative_samples_df

def prepare_data():
    generator = UserGenreProfileGenerator()
    user_profile = generator.build_profile(False)
    user_profile_matrix = user_profile.pivot(index='user_id', columns='genres', values='normalized_rating').fillna(0).reset_index()

    used_genres = [col for col in user_profile_matrix.columns if col not in ['user_id']]    
    
    raw_music_data = pandas.DataFrame(get_music_data(), columns=['track_id', 'genres'])
    raw_music_data['genres'] = raw_music_data['genres'].apply(ast.literal_eval)
    music_data_exploded = raw_music_data.explode('genres')
    music_data_exploded['value']=1
    music_data_exploded = music_data_exploded.drop_duplicates(subset=['track_id', 'genres'])
    
    song_genre_matrix = music_data_exploded.pivot(index='track_id', columns='genres', values='value').fillna(0).reset_index()
    interaction_data = pandas.DataFrame(get_interaction_data(), columns=['user_id', 'track_id', 'relevant'])

    for genre in used_genres:
        if genre not in song_genre_matrix.columns:
            song_genre_matrix[genre] = 0

    final_columns = ['track_id'] + used_genres
    song_genre_matrix = song_genre_matrix[final_columns]

    relevant_interactions = interaction_data[interaction_data['user_id'].isin(user_profile_matrix['user_id'])]
    negative_samples = generate_negative_samples(relevant_interactions, user_profile_matrix['user_id'].values, song_genre_matrix['track_id'].values)

    combined_data = relevant_interactions
    #combined_data = pandas.concat([relevant_interactions, negative_samples]).reset_index(drop=True)

    return user_profile_matrix, song_genre_matrix, combined_data

def build_model(num_genres):

    user_input = tf.keras.Input(shape=(num_genres,), name='user_profile')
    song_input = tf.keras.Input(shape=(num_genres,), name='song_genre')

    dot_product = tf.keras.layers.Dot(axes=1)([user_input, song_input])
    prediction = tf.keras.layers.Activation('sigmoid')(dot_product)

    model = tf.keras.Model(inputs=[user_input, song_input], outputs=prediction)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 

    return model

def build_interaction_labels(user_data, music_data, interaction_data):
    user_ids = user_data['user_id']
    track_ids = music_data['track_id']

    combinations = pandas.DataFrame({'user_id': user_ids.repeat(len(track_ids)),
                                 'track_id': list(track_ids) * len(user_ids)})
    
    interactions = pandas.merge(combinations, interaction_data, on=['user_id', 'track_id'], how='left')
    
    labels = interactions['relevant'].notnull().astype(int).values
    
    return labels


def evaluate_model(model, user_profiles, music_data, interaction_labels):
    labels = build_interaction_labels(user_profiles, music_data, interaction_labels)
    model.fit([user_profiles, music_data], labels, epochs=10, batch_size=1, validation_split=0.1)
    pass

def base_content_based_filtering_deep_learning():
    user_profile, music_data, interaction_data = prepare_data()
    model = build_model(user_profile.shape[1])
    evaluate_model(model, user_profile, music_data, interaction_data)
