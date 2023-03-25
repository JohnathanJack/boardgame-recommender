# boardgame-recommender

## Project/Goals
The goal of this project is to create a simple recommender system for the boardgamegeek dataset and evolutionize it into a modern hybrid recommender system. A moderm hybrid recommender system is one that utilizes collaboration and content based filtering features. An API utilizing Flask will be created to test pilot the deployment of the final model for educational purposes. 

## Process
1. View the dataset and choose the csv files that were relevant to recommender systems
2. Performed EDA by looking into the dispersion of ratings, the number of games published each year, most popular game, etc...
3. Create a content based recommender model using cosine similarity between each game's description
4. Create a collaborative filtering model using the surprise package on user-user ratings
5. Create a similar collaborative filtering model but using tensor flow
6. Improved upon the tensorflow model by elevating it with a neural network, creating a neural collaborative filtering model
7. Combined some content based features and the neural collaborative filtering model to a hybrid recommender model as the final model
8. The final model was saved and utilized in a Flask API for deployment

## Results
Out of all the models created, the model with the lowest RMSE was the hybrid recommender system. The hybrid model was successfully deployed locally and is able to return the top n recommendations for each given user. Each model had different recommendations for the same user which makes it hard to determine if there is a more correct model between all of them. This could be resolved with user feedback and utilizing A/B testing. 

## Challenges
Unable to test the recommender systems as it requires feedback from each user in the form of clicks, likes, purchases or survey. This made it difficult to determine if the recommender system is working correctly besides looking at the RMSE value. 


## Future Goals
Continue experimenting with adding more custom features and differnt hidden layers and see if those results in better recommendations for every user. 
