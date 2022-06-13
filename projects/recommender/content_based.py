import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
# https://www.projectpro.io/article/recommender-systems-python-methods-and-algorithms/413
pd.options.display.max_columns = None
pd.options.display.width = 1000
pd.options.display.max_colwidth =  1000

credits=pd.read_csv("projects/recommender/tmdb_5000_credits.csv")[['movie_id', 'cast', 'crew']].rename( columns={'movie_id': 'id'})
movies=pd.read_csv("projects/recommender/tmdb_5000_movies.csv")

movies = movies.merge(credits, on='id')
movies["overview"] =  movies["overview"].fillna('')

# ? need to dig deeper to understand how soup sould be built.
def create_soup(x):
 return ''.join(x['keywords']) + '' + ''.join(x['genres']) + '' + ''.join(x['overview'])
movies['soup']= movies.apply(create_soup, axis=1)

# Creating a TF-IDF Vectorizer
tfidf = TfidfVectorizer (stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['soup'])
tfidf_matrix.shape
# Calculating the Cosine Similarity – The Dot Product of Normalized Vectors
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
# Construct a reverse map of indices and movie titles
indices = pd.Series(movies.index, index=movies['title' ]).drop_duplicates()

#  Make a Recommendation
def get_recommendations(title, cosine_sim = cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]
    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list (enumerate (cosine_sim[idx]))
    # Sort the movies based on the similarity scores
    sim_scores = sorted (sim_scores, key=lambda x: x[1], reverse=True)
    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    # Return the ton 19 most similar movies
    return movies['title'].iloc[movie_indices]

get_recommendations(title = 'The Avengers', cosine_sim = cosine_sim)