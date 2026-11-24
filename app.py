
import streamlit as st
import pickle

# Page configuration
st.set_page_config(
    page_title="Netflix Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# Load saved KNN model and movie data
with open("movies_list.pkl", "rb") as f:
    movies = pickle.load(f)

with open("knn_model.pkl", "rb") as f:
    knn_model = pickle.load(f)

with open("genre_features.pkl", "rb") as f:
    genre_features = pickle.load(f)

movies_list = movies["title"].values


# KNN Recommendation Function
def recommend(movie_name):

    movie_index = movies[
        movies["title"].str.lower() == movie_name.lower()
    ].index

    if len(movie_index) == 0:
        return []

    index = movie_index[0]

    # Find nearest movies using KNN
    distances, indices = knn_model.kneighbors(
        genre_features[index].reshape(1, -1)
    )

    recommendations = []

    for i, distance in zip(indices[0][1:], distances[0][1:]):

        similarity_score = max(
            0.0,
            min(1.0, 1 - float(distance))
        )

        recommendations.append({
            "title": movies.iloc[i]["title"],
            "score": similarity_score,
            "genre": movies.iloc[i]["genre"]
        })

    return recommendations


# Custom Cinematic UI
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0b1020, #17132b, #101827);
    color: #f5f5f5;
}

h1 {
    color: #ffffff !important;
    font-weight: 800 !important;
}

h2, h3 {
    color: #f3f4f6 !important;
}

.stButton > button {
    background: linear-gradient(90deg, #e50914, #9b1c31);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.65rem 1rem;
    font-weight: 700;
    transition: 0.2s ease;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #ff2430, #c02645);
    color: white;
    border: none;
    transform: translateY(-2px);
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255, 255, 255, 0.04);
    border-radius: 14px;
    border-color: rgba(255, 255, 255, 0.14);
}

div[data-testid="stSelectbox"] {
    background: transparent;
}
</style>
""", unsafe_allow_html=True)


# Main UI
st.markdown("""
<div style="padding:35px 28px;
border-radius:20px;
background:linear-gradient(120deg,#25102f,#501525,#171c35);
border:1px solid rgba(255,255,255,0.14);
margin-bottom:25px;">

<p style="color:#ff6975;font-size:13px;font-weight:bold;
letter-spacing:3px;margin-bottom:10px;">
YOUR PERSONAL MOVIE GUIDE
</p><h1 style="color:white;font-size:38px;margin:0 0 12px 0;">
🎬 Discover Your Next Favourite Movie
</h1><p style="color:#e0dce7;font-size:17px;
max-width:750px;line-height:1.6;">
Find movies similar to the ones you love.
Explore recommendations powered by Machine Learning
using the K-Nearest Neighbors algorithm.
</p></div>
""", unsafe_allow_html=True)


# Movie Selection
selected_movie = st.selectbox(
    "🔍 Choose your favourite movie:",
    movies_list
)


# Show Recommendations
if st.button("🎯 Show Recommendations", use_container_width=True):

    recommendations = recommend(selected_movie)

    if not recommendations:
        st.warning("No recommendations found!")

    else:

        st.subheader(f"🎬 Movies similar to {selected_movie}")

        st.caption(
            "Recommended movies based on genre features using KNN"
        )

        # Display 3 cards per row
        for start in range(0, len(recommendations), 3):

            row_movies = recommendations[start:start + 3]
            columns = st.columns(3)

            for col, movie in zip(columns, row_movies):

                with col:

                    with st.container(border=True):

                        st.markdown(
                            f"### 🎬 {movie['title']}"
                        )

                        if movie.get("genre"):
                            st.write(
                                f"🎭 **Genre:** {movie['genre']}"
                            )

                        score = movie["score"]

                        st.write("📊 **Genre Similarity**")

                        st.progress(score)

                        st.caption(
                            f"Similarity: {score * 100:.2f}%"
                        )


st.divider()

st.caption(
    "Recommendations generated using KNN Machine Learning "
    "and One-Hot Encoded Genre Features."
)
