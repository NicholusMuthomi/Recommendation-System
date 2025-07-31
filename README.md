# Movie Recommendation System

## Overview

This project implements a collaborative filtering-based movie recommendation system using the MovieLens 20M dataset. The system predicts user ratings for movies and generates personalized recommendations by analyzing user behavior and movie attributes. The project explores various machine learning techniques including matrix factorization, KNN-based approaches, and ensemble methods.

## Features

- **Data Processing**: Handles large-scale movie rating data with 20M+ ratings from 138K users
- **Exploratory Analysis**: Provides insights into rating distributions, user behavior, and movie popularity
- **Multiple Algorithms**:
  - Baseline rating prediction
  - User-user and item-item collaborative filtering
  - Matrix Factorization (SVD, SVD++)
  - SlopeOne algorithm
  - XGBoost ensemble model
- **Recommendation Generation**: Produces personalized movie recommendations for users
- **Evaluation Metrics**: Uses RMSE and MAPE to assess model performance

## Installation

### Prerequisites

- Python 3.x
- Required libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `surprise`, `xgboost`, `fuzzywuzzy`

### Steps

1. Download the MovieLens 20M dataset from https://grouplens.org/datasets/movielens/20m/
2. Clone the repository (if applicable):
   ```bash
   git clone <repository_url>
   ```
3. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the analysis script:
   ```bash
   python movie_recommendation_system.py
   ```

## Dataset

The dataset contains:
- 20,000,263 ratings 
- 465,564 tag applications 
- 27,278 movies 
- 138,493 users 
- Data collected between January 1995 and March 2015

Files used:
- `ratings.csv`: User-movie ratings with timestamps
- `movies.csv`: Movie titles and genres

## Methodology

1. **Data Preparation**:
   - Merged rating and movie data
   - Created train-test split (80-20) based on timestamp
   - Handled sparse user-item matrices

2. **Feature Engineering**:
   - Extracted global, user, and movie average ratings
   - Computed similarity matrices (user-user and item-item)
   - Generated features from top similar users/items

3. **Modeling Approaches**:
   - **Baseline Models**: Global average, user average, movie average
   - **Neighborhood Methods**: User-user and item-item collaborative filtering
   - **Matrix Factorization**: SVD and SVD++ with implicit feedback
   - **Ensemble Method**: XGBoost combining all features

4. **Evaluation**:
   - RMSE (Root Mean Squared Error)
   - MAPE (Mean Absolute Percentage Error)

## Results

### Model Performance Comparison

| Model               | Train RMSE | Test RMSE |
|---------------------|------------|-----------|
| XGBoost (13 features)| 0.827      | 0.873     |
| BaselineOnly        | 0.876      | 0.892     |
| KNNBaseline (User)  | 0.852      | 0.887     |
| KNNBaseline (Item)  | 0.853      | 0.886     |
| SlopeOne            | 0.884      | 0.884     |
| SVD                 | 0.843      | 0.880     |
| SVDpp               | 0.837      | 0.879     |

### Key Findings

1. Matrix factorization techniques (SVD/SVD++) performed best among individual models
2. The ensemble XGBoost model combining all features achieved the lowest test RMSE (0.873)
3. User average rating was the most important feature for prediction
4. The system can effectively recommend movies with predicted ratings close to actual user preferences

## Usage

To generate recommendations for a user:

```python
# Get recommendations for user ID 42
recommendations = Generate_Recommendated_Movies(42)
print(recommendations)
```

Sample output:
```
   Movie_Id                     title                        genres  Predicted_Rating
0      318   Shawshank Redemption, The  Crime|Drama                    4.8
1      858   Godfather, The             Crime|Drama                    4.7
2       50   Usual Suspects, The        Crime|Mystery|Thriller         4.6
...
```

## Business Applications

1. **Personalized Recommendations**: Improve user engagement by suggesting relevant movies
2. **Rating Prediction**: Estimate how users would rate unseen movies
3. **Content Discovery**: Help users find niche content matching their preferences
4. **Inventory Management**: Identify popular and niche movies for better stocking

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For inquiries or feedback, please reach out to [Email Address](muthominicholus22@gmail.com).
