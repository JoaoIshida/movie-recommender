# Movie Recommendation System

## Overview

This project implements a movie recommendation system using Python, leveraging various libraries to analyze and recommend movies based on their titles and combined features (genres, keywords, and overview). The system is built with a focus on utilizing text-based data to generate relevant movie suggestions.

## Libraries and Functions Used

- **Pandas**:

  - Used for data manipulation and analysis.
  - It helped in loading the dataset, filling missing values, and combining multiple text features into a single column.
  - Documentation: [Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/)

- **Scikit-learn**:

  - A key library for machine learning in Python.
  - I utilized `TfidfVectorizer` to convert text data into numerical vectors. This transformation is crucial for calculating the similarity between different movies based on their features.

  ```python
  tfidf_vector = TfidfVectorizer(stop_words='english')
  tfidf_matrix = tfidf_vector.fit_transform(movies_df['combined_features'])
  ```

The cosine_similarity function from Scikit-learn computes the cosine similarity between the vectors, which helps determine how similar the movies are to each other.

Documentation: [Scikit-learn Documentation](https://scikit-learn.org/stable/)

### Difflib:

- Used to calculate the similarity ratio between movie titles, allowing the system to recommend movies with similar titles.
  Documentation: [Difflib Documentation](https://docs.python.org/3/library/difflib.html)

### Data Cleaning:

- Before processing the data, the initial dataset movies.csv underwent a cleaning phase to ensure its usability:

### Filling missing values:

- Any missing values in the dataset were filled with empty strings to prevent issues during feature combination and vectorization.

### Column Selection:

- Relevant columns were filtered to retain only the necessary features for the recommendation system as I saw fit for my beginner learning and future improvements. This helps streamline the dataset for the further processing.

### Combining Features:

- Relevant features (genres, keywords, and overview) are combined into a single string for each movie, enabling a comprehensive analysis.

### Vectorization:

- The combined features were converted into numerical vectors using TfidfVectorizer, which transforms the text data while ignoring common English stop words.

### Cosine Similarity Calculation:

- The cosine similarity matrix was computed to evaluate the similarity between movies based on their features.

### Recommendation Generation:

- The recommend_movies function identified the most similar movies based on both content similarity and title similarity, providing a ranked list of recommendations. Later sorted by similarity.

### Challenges Faced

- Data Quality: Initial dataset had missing values, requiring preprocessing to ensure the recommendation system functions effectively.
- Similarity Calculation: Balancing the weight of title similarity and content similarity to generate relevant recommendations required careful tuning.
- Learnings: Knowing how vectorization works, starting the project from the beginning with just the basic knowledge with pandas.

### Future Improvements

- Enhanced Features: Including more features such as cast, director, and release date to provide richer recommendations.
  Performance Optimization: Improving the efficiency of the recommendation algorithm to handle larger datasets.
  Application: Turn this system into an application or a better code with flask to be implemented in an App.

## References

[Understanding TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)

Data from [Kaggle](https://www.kaggle.com/datasets/harshshinde8/movies-csv)

## Conclusion

This project provided valuable insights into building a recommendation system using text analysis techniques. It demonstrated the effectiveness of using machine learning libraries in Python for processing and analyzing data to generate meaningful recommendations.
