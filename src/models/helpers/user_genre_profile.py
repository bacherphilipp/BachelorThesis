import ast

from sklearn.preprocessing import MinMaxScaler
from src.data_retrieval.dbconnect import get_connection
import pandas

def get_data():
    """Retrieves data for the different models.

    Parameters:

    Returns:

    """
    cursor = get_connection()
    print('Fetching data for profile buidling...')

    cursor.execute('''WITH ranked_users AS (
                        SELECT 
                            users.user_id, 
                            users.usergroup, 
                            users.isbyms,
                            users.playcountmintrack, 
                            users.playcountmaxtrack,
                            ROW_NUMBER() OVER (PARTITION BY users.usergroup, users.isbyms ORDER BY users.user_id) AS rn
                        FROM users
                    ),
                    top_users AS (
                        SELECT 
                            user_id, 
                            usergroup, 
                            isbyms,
                            playcountmintrack,
                            playcountmaxtrack
                        FROM ranked_users
                        WHERE rn <= 100
                    ),
                    user_events_ratings AS (
                        SELECT 
                            tu.user_id, 
                            e.track_id,
                            tu.usergroup, 
                            tu.isbyms, 
                            ga.genres,
                            ROUND(((COUNT(*) OVER (PARTITION BY e.track_id, tu.user_id) - tu.playcountmintrack)::NUMERIC / 
                                (CASE 
                                    WHEN tu.playcountmaxtrack - tu.playcountmintrack = 0 THEN 1 
                                    ELSE tu.playcountmaxtrack - tu.playcountmintrack 
                                END)) * (1000 - 1) + 1, 2)::FLOAT AS rating
                        FROM top_users tu
                        JOIN events e ON tu.user_id = e.user_id
                        JOIN genreannotation ga ON e.track_id = ga.track_id
                    )
                    SELECT 
                        user_id, 
                        track_id,
                        rating,
                        usergroup, 
                        isbyms, 
                        genres
                    FROM user_events_ratings
                    WHERE rating > 1
                    ORDER BY usergroup, isbyms, user_id;
    ''')
    print('Data fetched...')
    return cursor.fetchall()

def normalize_ratings(single_user):
    scaler = MinMaxScaler(feature_range=(1,1000))
    ratings = single_user['average_rating'].values.reshape(-1, 1)
    single_user['normalized_rating'] = scaler.fit_transform(ratings).flatten()
    return single_user

def build_profile(raw_data):
    raw_data['genres'] = raw_data['genres'].apply(ast.literal_eval)
    data_exploded = raw_data.explode('genres')
    genre_counts = data_exploded['genres'].value_counts()
    data_exploded['genre_weight'] = data_exploded['genres'].apply(lambda x: genre_counts[x])

    data_exploded['weighted_rating'] = data_exploded['rating'] * data_exploded['genre_weight']
    
    average_rating_by_genre = data_exploded.groupby(['user_id', 'genres'])['weighted_rating'].mean().reset_index(name='average_rating')

    average_rating_by_genre.reset_index(drop=True, inplace=True)

    user_genre_normalized = average_rating_by_genre.groupby('user_id', as_index=False).apply(normalize_ratings)

    user_genre_normalized = user_genre_normalized.reset_index(drop=True)

    print(user_genre_normalized.sort_values(by=['user_id', 'normalized_rating'], ascending=True).to_string())



def base_build_user_profile():
    data = get_data()
    raw_data = pandas.DataFrame(data, columns=['user_id', 'item_id', 'rating', 'usergroup', 'isbyms', 'genres' ])
    pandas.options.display.float_format = '{:.6f}'.format
    build_profile(raw_data)
    
