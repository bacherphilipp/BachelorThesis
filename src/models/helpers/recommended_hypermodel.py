from kerastuner import HyperModel
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Dense, Concatenate
from tensorflow.keras.models import Model

class RecommenderHyperModel(HyperModel):
    def __init__(self, num_users, num_tracks):
        self.num_users = num_users
        self.num_tracks = num_tracks

    def build(self, hp):
        user_input = Input(shape=(1,), name='user_input')
        track_input = Input(shape=(1,), name='track_input')

        embedding_size = hp.Choice('embedding_size', values=[50, 100, 150])

        user_embedding = Embedding(input_dim=self.num_users, output_dim=embedding_size, name='user_embedding')(user_input)
        track_embedding = Embedding(input_dim=self.num_tracks, output_dim=embedding_size, name='track_embedding')(track_input)

        user_vector = Flatten(name='flatten_user')(user_embedding)
        track_vector = Flatten(name='flatten_track')(track_embedding)

        concat = Concatenate()([user_vector, track_vector])

        dense_units_1 = hp.Int('dense_units_1', min_value=32, max_value=256, step=32)
        dense_units_2 = hp.Int('dense_units_2', min_value=32, max_value=256, step=32)
        dense_units_3 = hp.Int('dense_units_3', min_value=32, max_value=256, step=32)
        
        dense1 = Dense(units=dense_units_1, activation='relu')(concat)
        dense2 = Dense(units=dense_units_2, activation='relu')(dense1)
        dense3 = Dense(units=dense_units_3, activation='relu')(dense2)
        
        output = Dense(1, activation='linear')(dense3)

        model = Model(inputs=[user_input, track_input], outputs=output)
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

        return model
    
    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Choice("batch_size", [128, 256]),
            **kwargs,
    )