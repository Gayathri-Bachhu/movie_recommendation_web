from flask import Flask, render_template, request
import pandas as pd
from movie_recommendation_system import load_data, clean_data, content_based_recommender

app = Flask(__name__)

# Load and prepare data
df = clean_data(load_data("movies_metadata.csv"))
get_recommendations, indices = content_based_recommender(df)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    movie_title = request.form["title"]
    recommendations = get_recommendations(movie_title)

    if recommendations is None:
        return render_template("index.html", movie_title=movie_title, recommendations=[])

    result = [(row['title'], f"{row['similarity_score']:.2f}") for _, row in recommendations.iterrows()]
    return render_template("index.html", movie_title=movie_title, recommendations=result)

if __name__ == "__main__":
    app.run(debug=True)
