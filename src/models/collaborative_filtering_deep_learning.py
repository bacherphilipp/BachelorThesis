from src.models.helpers.recommended_hypermodel import RecommenderHyperModel
from src.data_retrieval.dbconnect import get_connection
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Dense, Concatenate
from tensorflow.keras.models import Model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from kerastuner.tuners import RandomSearch
import tensorflow as tf
from math import sqrt
import numpy as np
import pandas 
import configparser

def get_data():
    """Retrieves data for the different models.

    Parameters:

    Returns:

    """
    cursor = get_connection()
    print('Fetching data...')

    config = configparser.ConfigParser()
    config.read('config/settings.ini')
    data_record_limit = config.getint('general', 'data_record_limit', fallback=0)
    limit = '' if data_record_limit == 0 else ' limit ' + str(data_record_limit)

    cursor.execute('''select users.user_id, track_id,
                    round((( count(*) - playcountmintrack )::numeric / ( case when playcountmaxtrack - playcountmintrack = 0 then 1 else playcountmaxtrack - playcountmintrack end ) )* ( 1000 - 1) + 1 , 2)::float as rating,
                    users.usergroup, users.isbyms
                    from users join events on users.user_id = events.user_id
                    group by users.user_id, track_id''' + limit)
    print('Data fetched...')
    return cursor.fetchall()

def prepare_data(raw_data):
    """Prepares data for the model

    Parameters:
    raw_data (DataFrame): user-item relation data set

    Returns:
    DataFrame
    """
    user_encoder = LabelEncoder()
    track_encoder = LabelEncoder()

    raw_data['user_id'] = user_encoder.fit_transform(raw_data['user_id'])
    raw_data['item_id'] = track_encoder.fit_transform(raw_data['item_id'])

    raw_data['rating'] = raw_data['rating'] / 1000.0

    return raw_data

def evaluate_byms_groups(model, test_data):
    """Evaluates the results of the DL model by usergroup. Calculates RMSE and MAE

    Parameters:
    model (Model): the trained Model
    test_data (DataFrame): the test set portion of the useritem relation set

    Returns:
    
    """

    groups = dict()
    groups['all'] = [-1]
    groups['ms'] = test_data[test_data["isbyms"] == 0][['user_id', 'item_id', 'rating']].values.tolist()
    groups['byms'] = test_data[test_data["isbyms"] == 1][['user_id', 'item_id', 'rating']].values.tolist() 
    groups['UGR0'] = test_data[test_data["usergroup"] == 0][['user_id', 'item_id', 'rating']].values.tolist() 
    groups['UGR1'] = test_data[test_data["usergroup"] == 1][['user_id', 'item_id', 'rating']].values.tolist() 
    groups['UGR2'] = test_data[test_data["usergroup"] == 2][['user_id', 'item_id', 'rating']].values.tolist()
    groups['UGR3'] = test_data[test_data["usergroup"] == 3][['user_id', 'item_id', 'rating']].values.tolist() 

    for group_name, data in groups.items():
        if group_name != 'all':  
            if data:
                user_ids = np.array([pair[0] for pair in data])
                item_ids = np.array([pair[1] for pair in data])

                user_ids = user_ids.reshape(-1, 1) 
                item_ids = item_ids.reshape(-1, 1) 

                predictions = model.predict([user_ids, item_ids])

                predictions = [item for sublist in predictions.tolist() for item in sublist]
   
                true_ratings = [pair[2]*1000 for pair in data]

                pred_ratings =  [pred * 1000 for pred in predictions]

                print(f'RMSE for {group_name}:')
                print( np.sqrt(np.mean((np.array(true_ratings)-np.array(pred_ratings))**2)))

                print(f'MAE for {group_name}:')
                print(np.mean(np.abs(np.array(true_ratings) - np.array(pred_ratings))))

            else:
                print(f'No data to predict for group {group_name}')

def evaluate_deep_learning(useritem_data):
    """Evaluates the deep learning model with the parameters specified in the corresponding .ini.

    Parameters:
    useritem_data (DataFrame): The useritem relation dataset used to train the model

    Returns:
    
    """
    config = configparser.ConfigParser()
    config.read('config/hyperparameters/CBF_DL_hyperparameters.ini')

    num_users = useritem_data['user_id'].nunique()  
    num_tracks = useritem_data['item_id'].nunique() 

    train, test = train_test_split(useritem_data, test_size=0.2, random_state=42)

    user_input = Input(shape=(1,), name='user_input')
    track_input = Input(shape=(1,), name='track_input')

    embedding_size = config.getint('hyperparameters', 'embedding_size', fallback=50 )

    user_embedding = Embedding(input_dim=num_users, output_dim=embedding_size, name='user_embedding')(user_input)
    track_embedding = Embedding(input_dim=num_tracks, output_dim=embedding_size, name='track_embedding')(track_input)

    user_vector = Flatten(name='flatten_user')(user_embedding)
    track_vector = Flatten(name='flatten_track')(track_embedding)

    concat = Concatenate()([user_vector, track_vector])

    dense = Dense(256, activation='relu')(concat)
    dense = Dense(128, activation='relu')(dense)
    dense = Dense(64, activation='relu')(dense)
    dense = Dense(32, activation='relu')(dense)
    output = Dense(1, activation='linear')(dense) 

    model = Model(inputs=[user_input, track_input], outputs=output)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae']) 

    X_train = [train.user_id, train.item_id]
    y_train = train.rating
    X_test = [test.user_id, test.item_id]
    y_test = test.rating

    batch_size = config.getint('hyperparameters', 'batch_size', fallback=32 )
    epochs = config.getint('hyperparameters', 'epochs', fallback=15 )

    model.fit(X_train, y_train, batch_size, epochs, validation_data=(X_test, y_test))

    predictions = model.predict(X_test) 

    rmse = sqrt(mean_squared_error(y_test.values, predictions))
    print("Root Mean Squared Error (RMSE):", rmse*1000)

    mae = mean_absolute_error(y_test.values, predictions)
    print("Mean Absolute Error (MAE):", mae*1000)

    evaluate_byms_groups(model, test)

def evaluate_deep_learning_hypermodel(useritem_data):
    """Evaluates the deep learning model with the using a RandomSearch to try various hyperparameters

    Parameters:
    useritem_data (DataFrame): The useritem relation dataset used to train the model

    Returns:

    """

    num_users = useritem_data['user_id'].nunique()  
    num_tracks = useritem_data['item_id'].nunique() 

    hypermodel = RecommenderHyperModel(num_users, num_tracks)

    train, test = train_test_split(useritem_data, test_size=0.2, random_state=42)
    
    X_train = [train.user_id, train.item_id]
    y_train = train.rating
    X_test = [test.user_id, test.item_id]
    y_test = test.rating

    tuner = RandomSearch(
        hypermodel,
        objective='val_mae',
        max_trials=30,
        executions_per_trial=1,
        directory='tuner_results',
        project_name='recommender'
    )

    tuner.search(x=X_train, y=y_train, epochs=20, validation_data=(X_test, y_test),callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=3)])


def base_collaborative_filtering_deep_learning(use_tuner):
    """Entry function for collaborative filtering using deep learning.

    Parameters:

    Returns:

    """
    config = configparser.ConfigParser()
    config.read('config/hyperparameters/CBF_DL_hyperparameters.ini')

    tf.get_logger().setLevel(config.getint('general', 'log_level', fallback=0))

    data = get_data()
    raw_data = pandas.DataFrame(data, columns=['user_id', 'item_id', 'rating', 'usergroup', 'isbyms' ])

    useritem_data = prepare_data(raw_data)
    if use_tuner:
        evaluate_deep_learning_hypermodel(raw_data)
    else:
        evaluate_deep_learning(useritem_data)
    


