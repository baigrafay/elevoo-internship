import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split

# Load data
ratings = pd.read_csv('Datasets/ml-100k/u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])
movies = pd.read_csv('Datasets/ml-100k/u.item', sep='|', encoding='latin-1', 
                    names=['movie_id', 'title'] + [f'col_{i}' for i in range(22)], usecols=['movie_id', 'title'])
users = pd.read_csv("Datasets/ml-100k/u.user", sep="|", names=["user_id", "age", "gender", "occupation", "zip_code"])

# Split ratings into train/test (e.g., 80/20) by user to preserve user distribution
train_list = []
test_list = []

for uid, group in ratings.groupby('user_id'):
    train_grp, test_grp = train_test_split(group, test_size=0.2, random_state=42)
    train_list.append(train_grp)
    test_list.append(test_grp)

train_ratings = pd.concat(train_list)
test_ratings = pd.concat(test_list)

# Build user-item matrix from train only
user_item = train_ratings.pivot(index='user_id', columns='movie_id', values='rating')

# Recompute similarity matrices based on train data
user_sim = cosine_similarity(user_item.fillna(0))
user_sim_df = pd.DataFrame(user_sim, index=user_item.index, columns=user_item.index)

item_sim = cosine_similarity(user_item.fillna(0).T)
item_sim_df = pd.DataFrame(item_sim, index=user_item.columns, columns=user_item.columns)

def recommend_user_based(user_id, k=5, n_rec=5):
    # Find similar users
    sim_users = user_sim_df.loc[user_id].sort_values(ascending=False)[1:k+1].index
    sim_ratings = user_item.loc[sim_users].mean().sort_values(ascending=False)
    seen = user_item.loc[user_id].dropna().index
    recs = sim_ratings.drop(seen).head(n_rec)
    return movies[movies['movie_id'].isin(recs.index)][['movie_id', 'title']]

# Item-based collaborative filtering
item_sim = cosine_similarity(user_item.fillna(0).T)
item_sim_df = pd.DataFrame(item_sim, index=user_item.columns, columns=user_item.columns)

def recommend_item_based(user_id, n_rec=5):
    user_ratings = user_item.loc[user_id].dropna()
    sim_subset = item_sim_df[user_ratings.index]  
    scores = sim_subset.dot(user_ratings) / sim_subset.sum(axis=1)
    scores = scores.drop(user_ratings.index, errors='ignore')
    recs = scores.sort_values(ascending=False).head(n_rec)
    return movies[movies['movie_id'].isin(recs.index)][['movie_id', 'title']]


# Matrix factorization (SVD)
def recommend_svd(user_id, n_rec=5, n_components=20):
    user_item_filled = user_item.fillna(0)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    latent = svd.fit_transform(user_item_filled)
    reconstructed = np.dot(latent, svd.components_)
    user_idx = user_item.index.get_loc(user_id)
    preds = pd.Series(reconstructed[user_idx], index=user_item.columns)
    seen = user_item.loc[user_id].dropna().index
    recs = preds.drop(seen).sort_values(ascending=False).head(n_rec)
    return movies[movies['movie_id'].isin(recs.index)][['movie_id', 'title']]

def precision_at_k_test(user_id, rec_func, k=5):
    actual = test_ratings[(test_ratings['user_id'] == user_id) & (test_ratings['rating'] >= 4)]
    actual_pos = set(actual['movie_id'])
    recs = rec_func(user_id, n_rec=k)
    rec_ids = set(recs['movie_id'])
    if not actual_pos:
        return np.nan
    return len(actual_pos & rec_ids) / k

user_ids = user_item.index

print("Precision@5 for all users:")
for user_id in user_ids:
    print("User-based recommendations:")
    print(recommend_user_based(user_id))
    print("Item-based recommendations:")
    print(recommend_item_based(user_id))
    print("SVD-based recommendations:")
    print(recommend_svd(user_id))
    p_user = precision_at_k_test(user_id, recommend_user_based)
    p_item = precision_at_k_test(user_id, recommend_item_based)
    p_svd = precision_at_k_test(user_id, recommend_svd)
    if not np.isnan(p_user):
        print(f"User {user_id}: User-based={p_user:.3f}, Item-based={p_item:.3f}, SVD={p_svd:.3f}")
