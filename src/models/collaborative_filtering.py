from src.data_retrieval.dbconnect import get_connection
from surprise import KNNBasic, NMF
import configparser
import pandas
from src.models.helpers.custom_gridsearch_cv import CustomGridSearchCV


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

def filter_data(df, beyms=None, group=None):
    """Currently not in use, can filter a DataFrame by different criteria

    Parameters:
    df (DataFrame): the useritem dataset to be filtered
    beyms (int): filter beyond mainstream or mainstream listeners
    group (int): filter by user group (0-3)

    Returns:
    DataFrame: the DataFrame with the applied filters

    """

    if beyms != None:
        df = df[df['isbyms'] == beyms]
    if group != None and beyms == 1:
        df = df[df['usergroup'] == group]
    
    return df[['user_id', 'item_id', 'rating']]

def print_results(gs_results):
    """Prints a models results

    Parameters:
    gs_results (list: gridsearch_results): the result list for each UGR and evaluation metric

    Returns:

    """
    print("Best results for each group:")
    for gsr in gs_results:
        print(f"Group: {gsr.group}")
        print(f"  Best MAE Score: {gsr.best_score['mae']}")
        print(f"  Best MAE Params: {gsr.best_params['mae']}")
        print(f"  Best RMSE Score: {gsr.best_score['rmse']}")
        print(f"  Best RMSE Params: {gsr.best_params['rmse']}\n")

def evaluate_KNNBasic(useritem_dataset, cbf_result_test=False):
    """Evaluates a KNNBasic model with the given dataset

    Parameters:
    useritem_dataset (DataFrame): the user-item relation dataset
    cbf_result_test (bool): Whether the model should be used to check on CBF results

    Returns:

    """
    config = configparser.ConfigParser()
    config.read('config/hyperparameters/KNNBasic_hyperparameters.ini')

    k_values = [int(x.strip()) for x in config.get('hyperparameters', 'k').split(',')]
    sim_names = [x.strip() for x in config.get('hyperparameters', 'sim_name').split(',')]
    
    verbose = config.getboolean('general', 'verbose', fallback=True)

    param_grid = {
        'k': k_values,
        'sim_options': {
            'name': sim_names
        },
        'verbose': [verbose]
    }
    print("Starting KNNBasic hyperparametertuning...")
    gs = CustomGridSearchCV(KNNBasic, param_grid, measures= ["rmse", "mae"], cv=5, n_jobs=-1, refit=True)
    gs.fit(useritem_dataset)

    if cbf_result_test:
        test_cbf_results(gs)
    else:
        print_results(gs.gridsearch_results)

def evaluate_nmf(useritem_dataset, cbf_result_test=False):
    """Evaluates a NMF model with the given dataset

    Parameters:
    useritem_dataset (DataFrame): the user-item relation dataset
    cbf_result_test (bool): Whether the model should be used to check on CBF results

    Returns:

    """

    config = configparser.ConfigParser()
    config.read('config/hyperparameters/NMF_hyperparameters.ini')

    n_factor_values = [int(x.strip()) for x in config.get('hyperparameters', 'n_factors').split(',')]
    n_epoch_values = [int(x.strip()) for x in config.get('hyperparameters', 'n_epochs').split(',')]
    
    verbose = config.getboolean('general', 'verbose', fallback=True)

    param_grid = {
            'n_factors': n_factor_values,
            'n_epochs': n_epoch_values,
            'biased': [True, False],
            'verbose': [verbose]
    }

    print("Starting NMF hyperparametertuning...")
    gs = CustomGridSearchCV(NMF, param_grid, measures= ["rmse", "mae"], cv=5, n_jobs=-1, refit=True)
    gs.fit(useritem_dataset)


    if cbf_result_test:
        test_cbf_results(gs)
    else:
        print_results(gs.gridsearch_results)

def test_cbf_results(gs_model):
    """Tests a stored CBF resultset against a CF model to check if the recommended items are relevant

    Parameters:
    gs_model (CustomGridSearchView): a CustomGridSearchView with a trained model

    Returns:

    """
    config = configparser.ConfigParser()
    config.read('config/settings.ini')

    cf_cbf_result_check_records = config.getint('general','cf_cbf_result_check_records', fallback=20)

    file_path = 'results.txt'


    data = []
    with open(file_path, 'r') as file:
        for line in file:
            user_id, item_id = line.strip().split()  
            data.append({'user_id': int(user_id), 'item_id': int(item_id), 'rating': 1})

    new_useritem_dataset = pandas.DataFrame(data)

    testset = list(new_useritem_dataset[['user_id', 'item_id', 'rating']].iloc[:cf_cbf_result_check_records].itertuples(index=False, name=None))
    testres = gs_model.test(testset)
    # for prediction in testres:
    #      print(f"User: {prediction.uid}, Item: {prediction.iid}, Actual Rating: {prediction.r_ui}, Estimated Rating: {prediction.est}")

    average_estimated_rating = sum(prediction.est for prediction in testres) / len(testres)
    print(f"Average rating for all: {average_estimated_rating}")

    batch_averages, liked_recommendations = calculate_averages(testres, 100, 100)
    print(liked_recommendations)
    print(batch_averages)

def calculate_averages(predictions, batch_size=100, batch_size_items=100):
    """Calculates the average ratings per user group, generated using the CBF results in results.txt

    Parameters:
    predictions (DataFrame): the predictions generated from the CF model using the predicted CBF items
    batch_size (int): specify how many users are in each usergroup
    batch_size_items (int): specify how many items each user has

    Returns:
    averages (list)
    """
    averages = []
    liked_recommendations = []
    for i in range(0, len(predictions), batch_size * batch_size_items):
        batch = predictions[i:i + batch_size * batch_size_items]
        first_ratings = [batch[i].est for i in range(0, len(batch), 1)]
        liked_recommendations.append(sum(1 for item  in first_ratings if item >= 500))
        averages.append(sum(first_ratings) / (batch_size * batch_size_items))
    return averages, liked_recommendations

def base_collaborative_filtering(mode, cbf_eval = False):
    """Entry function for collaborative filtering.

    Parameters:
    mode (str): specifies which model should be evaluated (can be KNNBasic or NMF)
    cbf_eval (bool): specifies whether a CBF result should be checked with the specified model

    Returns:

    """

    data = get_data()
    raw_data = pandas.DataFrame(data, columns=['user_id', 'item_id', 'rating', 'usergroup', 'isbyms' ])

    if mode == "KNNBasic":
        evaluate_KNNBasic(raw_data, cbf_eval)
        return True
    elif mode == "NMF":
        evaluate_nmf(raw_data, cbf_eval)
        return True
    else:
        print("Collaborative filtering: Invalid mode specified!")
        return False



