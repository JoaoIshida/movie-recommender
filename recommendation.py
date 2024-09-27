import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher

movies_df = pd.read_csv("filtered_movies.csv")

movies_df.fillna('', inplace=True)

# Combine relevant text features into a single string for each movie
movies_df['combined_features'] = (
    movies_df['genres'] + ' ' + 
    movies_df['keywords'] + ' ' + 
    movies_df['overview']
)

# text data into vectors
tfidf_vector = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vector.fit_transform(movies_df['combined_features'])

# cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def title_similarity(title1, title2):
    return SequenceMatcher(None, title1.lower(), title2.lower()).ratio()

# movies based on title
def recommend_movies(title, num_recommendations=10):
    try:
        # get match title
        movie_index = movies_df[movies_df['title'] == title].index[0]
    except IndexError:
        print("Movie not found in dataset")
        return []

    # pairwise similarity score of movies matching title chosen
    sim_scores = list(enumerate(cosine_sim[movie_index]))

    title_similarities = {}
    # adding title similarity scores
    for idx in range(len(movies_df)):
        if idx != movie_index:
            similarity_score = title_similarity(title, movies_df['title'].iloc[idx])
            title_similarities[idx] = similarity_score

    combined_scores = []
    for idx, content_score in sim_scores:
        if idx == movie_index:
            continue
        title_score = title_similarities.get(idx, 0)
        combined_score = (content_score + title_score) /2 
        combined_scores.append((idx, combined_score))

    combined_scores = sorted(combined_scores, key=lambda x: x[1], reverse=True)

    # Get the top movie indices and their scores
    movie_indices = [i[0] for i in combined_scores[:num_recommendations]]
    movie_scores = [i[1] for i in combined_scores[:num_recommendations]]

    return list(zip(movies_df['title'].iloc[movie_indices].tolist(), movie_scores))

recommendations = recommend_movies("Toy Story", num_recommendations=10)

# Displaying movie recommendations along with scores
for movie, score in recommendations:
    print(f"{movie}: {score}")
