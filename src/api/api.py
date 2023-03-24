from flask import jsonify, Flask, request
from flask_restful import Resource, Api, reqparse
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


app = Flask(__name__)
api = Api(app)

# Necessary dataframes to get top recommendations for a user
df = pd.read_csv('user_ratings.csv')
game_df = pd.read_csv('games.csv')

# Ensure all datatypes are the same 
df['Username'] = df['Username'].astype(str)
df['BGGId'] = df['BGGId'].astype(int)
df['Rating'] = df['Rating'].astype('float32')

# Encode the BGGId and Username to a certain value via a dictionary
user_ids = df['Username'].unique().tolist()
game_ids = df['BGGId'].unique().tolist()
user2user_encoded = {x: i for i, x in enumerate(user_ids)}
game2game_encoded = {x: i for i,x in enumerate(game_ids)}

def df_with_features(df1, df2):
    """
        Creates a dataframe with the necessary features (Username, BGGId, Rating, user, game, new_or_old, BayesAvgRating, BestPlayers, MfgPlaytime, NumUserRatings) for predictions

        Parameters:
            df1: Dataframe with features: YearPublished, BayesAvgRating, BestPlayers, MfgPlaytime, NumUserRatings
            df2: Dataframe with features: Username, BGGId, Rating

        Returns:
            A combined dataframe with the features: Username, BGGId, Rating, user, game, new_or_old, BayesAvgRating, BestPlayers, MfgPlaytime, NumUserRatings

    """

    feats = ['BGGId','new_or_old','BayesAvgRating','BestPlayers','MfgPlaytime','NumUserRatings']
    feats_df = df1[feats]
    combined_features_df = feats_df.merge(df2, on='BGGId')
    combined_features_df['user_encode'] = combined_features_df['Username'].map(user2user_encoded)
    combined_features_df['bggid_encode'] = combined_features_df['BGGId'].map(game2game_encoded)
    return combined_features_df

combined_features_df = df_with_features(game_df, df)

# Load the model
model = tf.keras.models.load_model('hybrid_model')

def user_recommendations(user, n_games,model):
    """
        Takes in a user, n_games and a model to returns their top n_games recommendations that they have not rated yet.

        Parameters:
            user(str): name of user
            n_games(int): number of games to show
            model: the model to use for predictions
            
        Returns:
            A list with each game as an index with a column for their estimated rating
    """
    # Remove the boardgames that the user have rated
    boardgame_ids = combined_features_df['BGGId'].unique()
    user_game_list = combined_features_df.loc[combined_features_df['Username']==user, 'BGGId']
    user_game_list = np.setdiff1d(boardgame_ids, user_game_list)

    # Create dataframe of unique games for the user and a dataframe of the boardgames features
    testset = [[user, bggid] for bggid in user_game_list]
    testset = pd.DataFrame(testset, columns=['Username','BGGId'])
    feats = ['BGGId','new_or_old','BayesAvgRating','BestPlayers','MfgPlaytime','NumUserRatings']
    feat_df = game_df[game_df['BGGId'].isin(user_game_list)][feats]
    # Combine the two dataframes to be used in prediction model
    testset = df_with_features(feat_df, testset)
    feats = ['new_or_old','BayesAvgRating','BestPlayers','MfgPlaytime','NumUserRatings']
    feats_to_scale = ['BayesAvgRating','MfgPlaytime','NumUserRatings']

    # Get training data and scale it to apply to the testset
    scaler = MinMaxScaler()
    X = combined_features_df[['user_encode','bggid_encode','new_or_old','BayesAvgRating','BestPlayers','MfgPlaytime','NumUserRatings']]
    y = combined_features_df[['Rating']]
    x_train, _, _, _ = train_test_split(X,y,test_size=0.2, random_state=0)
    x_train[feats_to_scale] = scaler.fit_transform(x_train[feats_to_scale])
    testset[feats_to_scale] = scaler.transform(testset[feats_to_scale])

    # Test model on the testset and determine the boardgames with the highest rating
    predictions = model.predict(x=[testset['user_encode'],testset['bggid_encode'],testset[feats]])
    # Reshaped the predictions array by Transposing it. 
    top_ratings_idx = predictions.T[0].argsort()[::-1][:n_games]
    bgg_ids = user_game_list[top_ratings_idx]
    bgg_name = [game_df.loc[game_df['BGGId'] ==id]['Name'].values[0] for id in bgg_ids]
    print(f'Top boardgames for {user} in order are: \n {bgg_name}')

class Predict(Resource):
    def post(self):
        json_data = request.get_json()
        df = pd.DataFrame(json_data.values(), index=json_data.keys()).transpose()
        prediction = user_recommendations(df.iloc[0].values[0], df.iloc[0].values[1], model)
        return jsonify({'prediction': prediction})

api.add_resource(Predict, '/predict')

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=5000)
