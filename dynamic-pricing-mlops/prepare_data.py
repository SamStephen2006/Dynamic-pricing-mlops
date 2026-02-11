import pandas as pd
import numpy as np
import os

# 1. Setup
np.random.seed(42)
n_orders = 5000
os.makedirs("data", exist_ok=True)

# 2. Generate Random Features (The "Inputs")
hours = np.random.randint(0, 24, n_orders)
weather = np.random.choice([0, 1, 2], n_orders, p=[0.7, 0.2, 0.1]) # 0=Clear, 1=Rain, 2=Storm
exam_season = np.random.choice([0, 1], n_orders, p=[0.8, 0.2])
workers = np.random.randint(10, 100, n_orders)

# 3. Calculate Surge (The "Target")
# Start with a base multiplier of 1.0 for everyone
surge = np.ones(n_orders)

# --- YOUR TASK: FILL IN THE LOGIC BELOW ---

# Rule 1: If weather is Rain (1), add 0.1 to surge. Else keep surge as is.
surge = np.where(weather == 1, surge + 0.1, surge)

# Rule 2: If weather is Storm (2), add 0.05 to surge. 
# Write the line below using np.where
surge = np.where(weather == 2, surge + 0.05, surge)
# <--- WRITE THIS LINE

# Rule 3: If it is Exam Season (1), add 0.3 to surge.
surge = np.where(exam_season == 1, surge + 0.3, surge)
# <--- WRITE THIS LINE

# Rule 4: If workers are less than 20, add 0.5 to surge.
surge = np.where(workers < 20, surge + 0.5, surge)
# <--- WRITE THIS LINE

# ------------------------------------------

# 4. Save to DataFrame
df = pd.DataFrame({
    'hour': hours,
    'weather': weather,
    'exam_season': exam_season,
    'workers': workers,
    'surge_multiplier': surge
})

# Add some random noise so the model doesn't memorize perfect rules (Real life is messy)
df['surge_multiplier'] += np.random.normal(0, 0.05, n_orders)
df['surge_multiplier'] = df['surge_multiplier'].round(2)

# Save
csv_path = "data/gig_pricing.csv"
df.to_csv(csv_path, index=False)
print(f" Data generated at {csv_path}")