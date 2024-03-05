from src.data_retrieval.dbconnect import get_connection
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import pandas
import numpy as np
import configparser
import os

cursor = None


class AttributeResults:
    def __init__(self,attributes):
        self.attributes = attributes
        self.rmse = list()
        self.mae = list()
        self.precision = list()
        self.recommended_songs = list()
        self.userid = 0
    def __str__(self):
        return f"Attributes: {self.attributes}\n Average rmse: {sum(self.rmse)/len(self.rmse) }\n Average mae: {sum(self.mae)/len(self.mae) }"
    
def get_music_data():
    """Retrieves music data containing track features. Can be limites via the data_record_limit setting

    Parameters:

    Returns:

    """
    print('Fetching music data...')
    config = configparser.ConfigParser()
    config.read('config/settings.ini')
    data_record_limit = config.getint('general', 'data_record_limit', fallback=0)
    limit = '' if data_record_limit == 0 else ' limit ' + str(data_record_limit)

    cursor.execute('''
        select
            track_id,
            COALESCE(danceability, 0) AS danceability,
            COALESCE(energy, 0) AS energy,
            COALESCE(key, 0) AS key,
            COALESCE(loudness*-1, 0) AS loudness,
            COALESCE(mode, 0) AS mode,
            COALESCE(speechiness, 0) AS speechiness,
            COALESCE(acousticness, 0) AS acousticness,
            COALESCE(instrumentalness, 0) AS instrumentalness,
            COALESCE(liveness, 0) AS liveness,
            COALESCE(valence, 0) AS valence,
            COALESCE(tempo, 0) AS tempo
        from
            tracks
        where  
            tracks.track_id in (select track_id from events)
       
    ''' + limit)
    print('Music data fetched...')
    return cursor.fetchall()

def get_user_data():
    """Retrieves user data. 100 users each (100 ms, 100 beys and 100 for each byms group)

    Parameters:

    Returns:

    """
    print('Fetching user data...')
    cursor.execute('''WITH ranked_users AS (
                        SELECT 
                            users.user_id,
                            users.usergroup,
                            users.isbyms,
                            AVG(danceability) AS avg_danceability,
                            AVG(energy) AS avg_energy,
                            AVG(key) AS avg_key,
                            AVG(loudness * -1) AS avg_loudness,
                            AVG(mode) AS avg_mode,
                            AVG(speechiness) AS avg_speechiness,
                            AVG(acousticness) AS avg_acousticness,
                            AVG(instrumentalness) AS avg_instrumentalness,
                            AVG(liveness) AS avg_liveness,
                            AVG(valence) AS avg_valence,
                            AVG(tempo) AS avg_tempo,
                            ROW_NUMBER() OVER (PARTITION BY users.isbyms,users.usergroup ORDER BY users.user_id) AS rn
                        FROM users 
                        JOIN events ON users.user_id = events.user_id 
                        JOIN tracks ON events.track_id = tracks.track_id 
                        GROUP BY users.user_id, users.usergroup
                        )
                        SELECT 
                        user_id,
                        avg_danceability,
                        avg_energy,
                        avg_key,
                        avg_loudness,
                        avg_mode,
                        avg_speechiness,
                        avg_acousticness,
                        avg_instrumentalness,
                        avg_liveness,
                        avg_valence,
                        avg_tempo
                        FROM ranked_users
                        WHERE rn <= 100
                        ORDER BY isbyms,usergroup,rn
                        LIMIT 500;
                        ''')
    print('User data fetched...')
    return cursor.fetchall()

def prepare_data(raw_music_data, raw_user_data):
    """Prepares the data for content based filtering. Features to be evaluated can be specified in .ini

    Parameters:
    raw_music_data (DataFrame): the music data retrieved from the DB
    raw_user_data (DataFrame): the user data retrieved from the DB
    
    Returns:
    music_data (DataFrame)
    user_data (DataFrame)
    column_powerset (list)
    
    """

    config = configparser.ConfigParser()
    config.read('config/hyperparameters/cbf_featureset.ini')


    pandas.options.display.float_format = '{:.6f}'.format
    scaler = MinMaxScaler(feature_range=(1,1000))

    features = [x.strip() for x in config.get('hyperparameters', 'features').split(',')]
    
    columns = []

    if len(features)>0:
        columns = features
    else:
        columns = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    column_powerset = powerset(columns)

    raw_music_data[columns] = scaler.fit_transform(raw_music_data[columns])
    raw_user_data[columns] = scaler.transform(raw_user_data[columns])

    return raw_music_data, raw_user_data, column_powerset

def powerset(attributes):
    """Builds a powerset of item features if generate_powerset=True

    Parameters:
    attributes (list): A list of items features
    
    Returns:
    (list)

    """
    config = configparser.ConfigParser()
    config.read('config/hyperparameters/cbf_featureset.ini')

    generate_powerset = config.getboolean('general', 'generate_powerset', fallback=False)

    if generate_powerset:
        res = [[]]
        for e in attributes:
            res += [sub + [e] for sub in res]
        return [elem for elem in res if elem != []]
    else:
        return [attributes]

def calc_mae_rmse(actual_attributes, recommended_attributes):
    """Calculates mae and rmse. The result only  tells us how close we got in actually matching the acoustic features. It doesn't measure if the user likes the recommended item.

    Parameters:
    actual_attributes (list): list of actual item attributes
    recommended_attributes (list): list of the recommended attributes
    
    Returns:
    rmse (float)
    mae (float)

    """

    mae_values = []
    mse_values = []
    for _, song_row in recommended_attributes.iterrows():
        mae = mean_absolute_error(actual_attributes, song_row)
        mse = mean_squared_error(actual_attributes, song_row)
        mae_values.append(mae)
        mse_values.append(mse)

    rmse = np.sqrt(np.mean(mse_values))
    mae = np.mean(mae_values)
    return rmse, mae

def calc_precision(user, recommended_songs):
    """Calculates the precision based on songs the user has listened to + songs of artists of songs that the user has listened to. 

    Parameters:
    user (int): the users id
    recommended_songs (list): list of songs recommended to the user
    
    Returns:
    precision (float)

    """
    user = str(int(user))
    tracks = ','.join(map(str, recommended_songs['track_id']))
    cursor.execute('select count(*) from tracks where artist_id in (select distinct artist_id from events where user_id = ' + user + ' and track_id in (' + tracks+ ')) and track_id in (' + tracks+ ')' )
    precision = cursor.fetchall()[0][0] / len(recommended_songs)
    return precision

def print_eval_results(result_list):
    """Function to print the results for the best feature combinations, sorted by precision. 

    Parameters:
    result_list (list): list of AttributeResults
    
    Returns:

    """

    average_precisions = [(results, sum(results.precision) / len(results.precision)) for results in result_list]
    sorted_results = sorted(average_precisions, key=lambda x: x[1], reverse=True)

    print('Top 10 Results (based on precision)')
    for res in sorted_results[:10]:
        print(res[0])
        print('Average precision ' + str(res[1]) + '\n')

def evaluate_content_based_filtering(music_data, user_data, column_powerset):
    """Evaluates content based filtering using a neighbor based approach. Hyperparameters can be tuned in the .ini.
    If you want to store the results to check them on a CF model, you can set store_to_file=True in the .ini.

    Parameters:
    music_data (DataFrame): preprocessed music data
    user_data (DataFrame): preprocessed user data
    column_powerset ([list]): powerset (if specified, otherwise just one list) of the feature attribues
    
    Returns:

    """
        
    config = configparser.ConfigParser()
    config.read('config/hyperparameters/cbf_featureset.ini')

    n_nearest_neighbors = config.getint('hyperparameters', 'n_nearest_neighbors', fallback=50)
    metric = config.get('hyperparameters', 'metric', fallback='cosine')

    store_to_file = config.getboolean('general', 'store_to_file', fallback=False)

    results_list = list()
    len_powerset = len(column_powerset)
    
    f = None
    if store_to_file:
        if os.path.exists('results.txt'):
            os.remove('results.txt')
        f = open('results.txt', 'a')

    for count, attribute_columns in enumerate(column_powerset):
        print('Evaluating for attributes: ' + ', '.join(attribute_columns))
        print('Progress:' + str(count) + '/' + str(len_powerset) + '\n')
        
        knn_model = NearestNeighbors(n_neighbors=n_nearest_neighbors, metric=metric)
        knn_model.fit(music_data[attribute_columns].values)
        
        attribute_results = AttributeResults(attribute_columns)

        user_ratings = {}
        for index, user_row in user_data.iterrows():
            _, indices = knn_model.kneighbors([user_row[attribute_columns]], n_nearest_neighbors)

            recommended_songs = music_data.iloc[indices[0]][['track_id'] + attribute_columns]
            user_ratings[user_row['user_id']] = recommended_songs

            r,m = calc_mae_rmse(user_row[attribute_columns], recommended_songs[attribute_columns])
            attribute_results.rmse.append(r)
            attribute_results.mae.append(m)
            attribute_results.userid=user_row['user_id']
            attribute_results.recommended_songs = recommended_songs
            p=calc_precision(user_row['user_id'], recommended_songs[['track_id']])

            if store_to_file:
                for track_id in recommended_songs['track_id']:
                    f.write(str(int(user_row['user_id'])) + ' ' + str(track_id) + '\n')
                
            attribute_results.precision.append(p)
        results_list.append(attribute_results)

    if store_to_file:
        f.close()

    print_eval_results(results_list)


def base_content_based_filtering():
    """Entry function for content based filtering.

    Parameters:

    Returns:

    """

    global cursor
    cursor = get_connection()

    music_data = get_music_data()
    user_data = get_user_data()

    raw_music_data = pandas.DataFrame(music_data, columns=['track_id', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'])
    raw_user_data = pandas.DataFrame(user_data, columns=['user_id', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'])

    music_data, user_data, column_powerset = prepare_data(raw_music_data, raw_user_data)
    evaluate_content_based_filtering(music_data, user_data, column_powerset)