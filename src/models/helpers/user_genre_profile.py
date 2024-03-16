import ast
from sklearn.preprocessing import MinMaxScaler
from src.data_retrieval.dbconnect import get_connection
import pandas

class UserGenreProfileGenerator:

    def __init__(self):
        pass

    def get_data(self, limit_rating=True):
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
                            WHERE rn <= 300
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
                       ''' + ("WHERE rating > 1" if limit_rating else "") +  '''
                        ORDER BY usergroup, isbyms, user_id
        ''')
        print('Data fetched...')
        return cursor.fetchall()

    def prepare_data(self):
        data = self.get_data()
        raw_data = pandas.DataFrame(data, columns=['user_id', 'item_id', 'rating', 'usergroup', 'isbyms', 'genres' ])
        pandas.options.display.float_format = '{:.6f}'.format
        return raw_data

    def normalize_ratings(self, single_user):
        scaler = MinMaxScaler(feature_range=(1,1000))
        ratings = single_user['average_rating'].values.reshape(-1, 1)
        single_user['normalized_rating'] = scaler.fit_transform(ratings).flatten()
        return single_user

    def build_profile(self, printProfile=False):
        raw_data = self.prepare_data()

        raw_data['genres'] = raw_data['genres'].apply(ast.literal_eval)
        data_exploded = raw_data.explode('genres')
        genre_counts = data_exploded['genres'].value_counts()
        data_exploded['genre_weight'] = data_exploded['genres'].apply(lambda x: genre_counts[x])

        data_exploded['weighted_rating'] = data_exploded['rating'] * data_exploded['genre_weight']
        
        average_rating_by_genre = data_exploded.groupby(['user_id', 'genres'])['weighted_rating'].mean().reset_index(name='average_rating')

        average_rating_by_genre.reset_index(drop=True, inplace=True)

        user_genre_normalized = average_rating_by_genre.groupby('user_id', as_index=False).apply(self.normalize_ratings)

        user_genre_normalized = user_genre_normalized.reset_index(drop=True)

        if printProfile:
            print(user_genre_normalized.sort_values(by=['user_id', 'normalized_rating'], ascending=True).to_string())

        user_genre_normalized = user_genre_normalized[user_genre_normalized['normalized_rating'] >= 250]
        return user_genre_normalized

