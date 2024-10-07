# Cricket Match Outcome Predictor

This project implements a predictive model for cricket match outcomes using ball-by-ball data and match-level information from the Indian Premier League (IPL). The model is built using Python, Pandas, and Scikit-learn, and predicts the likelihood of a team winning a match as the game progresses. The data includes team names, city, total runs, wickets left, current run rate (CRR), and required run rate (RRR).

## Features

- **Predict Match Outcomes**: Uses logistic regression to predict the probability of a team winning based on match conditions.
- **Progression Tracking**: Shows how win/loss probabilities evolve over the course of the second innings.
- **Real-time Visualization**: Graphs match progression, including runs scored per over, wickets lost, and win probabilities.
- **Flexible Data Handling**: Handles various team names, corrects for legacy IPL teams, and accounts for missing data using city-specific information.

## Data

- `deliveries.csv`: Contains ball-by-ball match data.
- `matches.csv`: Contains match-level data with details like team names, city, winner, and match ID.

## Workflow

### 1. **Data Loading**
   The `deliveries.csv` and `matches.csv` files are loaded into Pandas DataFrames.

### 2. **Data Cleaning and Preprocessing**
   - The data is filtered to include only IPL teams that are currently active.
   - Legacy team names like "Deccan Chargers" and "Delhi Daredevils" are updated to their current counterparts.
   - Matches impacted by the Duckworth-Lewis (DL) method are removed.

### 3. **Feature Engineering**
   - Calculates cumulative runs in the second innings and computes:
     - **Runs Left**
     - **Balls Left**
     - **Wickets Left**
     - **Current Run Rate (CRR)**
     - **Required Run Rate (RRR)**
   - Creates target labels indicating whether the batting team won the match.

### 4. **Modeling**
   - A logistic regression model is trained using a pipeline that performs one-hot encoding on categorical variables (`batting_team`, `bowling_team`, and `city`).
   - The model is trained on a portion of the dataset, while another portion is used for testing and validation.

### 5. **Prediction**
   - The model predicts win/loss probabilities for the batting team during the second innings, calculating the chances of winning at the end of each over.
   - The function `match_progression` is used to compute win probabilities and other metrics like runs scored per over and wickets lost in each over.

### 6. **Visualization**
   - A plot visualizes match progression, showing the win/loss probability and key match events like runs scored per over and wickets lost.
   
   Example:
   ```python
   temp_df, target = match_progression(delivery_df, 513, pipe)
   plt.plot(temp_df['end_of_over'], temp_df['win'], label='Win Probability', color='green')
   plt.plot(temp_df['end_of_over'], temp_df['lose'], label='Lose Probability', color='red')
   plt.bar(temp_df['end_of_over'], temp_df['runs_after_over'], label='Runs Scored')
   ```

### 7. **Model Persistence**
   - The trained model pipeline is saved as `pipe.pkl` for future predictions.

## Requirements

To run the project, install the following dependencies:

```bash
pip install numpy pandas seaborn matplotlib scikit-learn
```

## How to Use

1. Clone the repository and ensure you have the required dependencies installed.
2. Place the `deliveries.csv` and `matches.csv` files in the root folder.
3. Run the script to preprocess the data and train the model.
4. Use the `match_progression` function to predict the outcome of a match at the end of each over.
5. Visualize the match progression with the provided plotting code.

## Example

```python
import pickle

# Load the saved model pipeline
pipe = pickle.load(open('pipe.pkl', 'rb'))

# Predict match progression for a specific match_id
match_id = 513
temp_df, target = match_progression(delivery_df, match_id, pipe)

# Visualize the results
import matplotlib.pyplot as plt
plt.figure(figsize=(18,8))
plt.plot(temp_df['end_of_over'], temp_df['win'], color='green', label='Win Probability')
plt.plot(temp_df['end_of_over'], temp_df['lose'], color='red', label='Lose Probability')
plt.bar(temp_df['end_of_over'], temp_df['runs_after_over'], color='blue', label='Runs Scored')
plt.xlabel('End of Over')
plt.ylabel('Probability / Runs')
plt.title(f'Target: {target}')
plt.legend()
plt.show()
```

## License

This project is open source and free to use.
