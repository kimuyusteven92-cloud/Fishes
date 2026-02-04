import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# 1. MOCK DATA GENERATION (Replace with API data for live use)
# In a real scenario, you'd use 'nfl_data_py' or 'balldontlie' API
def generate_sports_data(league="NBA"):
    np.random.seed(42)
    games = 1000
    data = {
        'avg_pts_home': np.random.normal(110, 10, games),
        'avg_pts_away': np.random.normal(108, 10, games),
        'off_efficiency_home': np.random.uniform(100, 120, games),
        'def_efficiency_away': np.random.uniform(100, 120, games),
        'pace_factor': np.random.uniform(90, 105, games), # Crucial for O/U
        'rest_days_home': np.random.randint(1, 4, games),
        'is_back_to_back': np.random.randint(0, 2, games),
        'actual_total': []
    }
    
    # Logic: Total score is a function of efficiency and pace
    for i in range(games):
        base = (data['off_efficiency_home'][i] + data['def_efficiency_away'][i]) / 2
        total = (base * (data['pace_factor'][i] / 100)) * 2 + np.random.normal(0, 5)
        data['actual_total'].append(round(total, 1))
        
    return pd.DataFrame(data)

# 2. MODEL TRAINING
df = generate_sports_data()
X = df.drop('actual_total', axis=1)
y = df['actual_total']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 3. FORECASTING FUNCTION
def predict_total(home_eff, away_eff, pace, b2b):
    # Setup your matchup data
    matchup = pd.DataFrame([[110, 108, home_eff, away_eff, pace, 2, b2b]], 
                           columns=X.columns)
    prediction = model.predict(matchup)[0]
    
    print(f"--- 🏀 Predicted Total Score: {prediction:.2f} ---")
    print(f"Confidence Level: {model.score(X_test, y_test)*100:.2f}%")
    return prediction

# Example: High Pace NBA Game, Home Team on Back-to-Back
predict_total(home_eff=115.5, away_eff=112.0, pace=102.5, b2b=1)
