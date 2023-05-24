import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import PorterStemmer

# Load the movie dictionary
movies_dict = pickle.load(open('movies_dict.pkl', 'rb'))
new_df = pd.DataFrame.from_dict(movies_dict)

# Calculate cosine similarity matrix
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tag']).toarray()
similarity = cosine_similarity(vectors)

# Function to recommend movies
def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    recommended_movies = [new_df.iloc[i[0]].title for i in movies_list]
    return recommended_movies

# Main function
def main():
    st.title("Movie Recommendation System")

    # User input
    movie = st.text_input("Enter a movie name:")

    if st.button("Recommend"):
        if movie in new_df['title'].values:
            recommended_movies = recommend(movie)
            st.subheader("Recommended Movies:")
            for recommended_movie in recommended_movies:
                st.write(recommended_movie)
        else:
            st.write("Movie not found in the database.")

if __name__ == '__main__':
    main()
