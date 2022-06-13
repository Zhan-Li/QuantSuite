import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
# https://www.projectpro.io/article/recommender-systems-python-methods-and-algorithms/413
# read data
amzn_ratings= pd.read_csv('projects/recommender/ratings_Beauty.csv')
amzn_ratings.head()
amzn_ratings= amzn_ratings.sample(25000, random_state =  0)
# construct user-item matrix and fill NA as 0
new_df = amzn_ratings[['UserId', 'ProductId', 'Rating']].pivot(index = 'UserId', columns = 'ProductId').fillna(0)
new_df = new_df.droplevel(0, axis = 1)
# pick one user to find similar users
user_id = 'A1BWKPDQD4MTTR'
# the user's ratings on all products
user=new_df.loc[new_df.index == user_id].squeeze()
product_ratings = user.loc[user > 0].to_dict()
# get other users who has voted on the product id B00CJ01QT8
subset=new_df.loc[new_df['B00CJ01QT8'] > 0 ]
len(subset)
subset.index
# calculate user similarity score
res = {}
for user_id in subset.index:
    cos=cosine_similarity(user.values.reshape(1,-1), subset.loc[user_id].values.reshape(1,-1))
    res[user_id]=cos


# now we look for non-null voting from similar users
res = new_df.loc[(new_df.index == 'A1M7G0O4T9XANR') | (new_df.index== 'A35SV0ZC5ONKJ7')].transpose()\
    .iloc[1:,:]\
    .agg('sum', axis = 1)
res[res >0]
# Implementing a Model-Based Recommendation System

from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate
reader= Reader(rating_scale=(1, 5))
amzn_ratings.head()

data = Dataset.load_from_df(amzn_ratings[['UserId','ProductId', 'Rating']], reader)
svd = SVD()
cross_validate(svd, data, measures=['RMSE', 'MAE'],cv=5)

trainset= data.build_full_trainset()
svd.fit(trainset)

amzn_ratings[amzn_ratings['UserId']=='A281NPSIMI1C2R']

svd.predict('A281NPSIMI1C2R','B804WPHRZA')


for idx, i in enumerate([1,2,3]):
    print(idx, i)