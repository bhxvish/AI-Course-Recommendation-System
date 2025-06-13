import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

df = pd.read_csv("courses.csv")

vectorize = TfidfVectorizer()
matrix = vectorize.fit_transform(df["description"])
similarity = cosine_similarity(matrix)

def recommendation(selected_course):
    index = df[df["title"] == selected_course].index[0]
    scores = list(enumerate(similarity[index]))
    sorted_scores = sorted(scores, key = lambda x: x[1], reverse = True)
    recommend = [df["title"][i[0]] for i in sorted_scores[1:6]]
    return recommend

st.title("AI Course Recommendation System")
course_list = df["title"].tolist()
selected_course = st.selectbox("Choose which course you want to pursue", course_list)

if st.button("Recommend"):
    result = recommendation(selected_course)
    st.write("Recommened Courses:")
    for course in result:
        st.write(f"-{course}")