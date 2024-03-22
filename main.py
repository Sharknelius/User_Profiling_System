import csv
import random
import pandas as pd
from pprint import pprint
import numpy as np
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from surprise import Reader, Dataset
from faker import Faker

"""
fake = Faker()
roomData = []

for _ in range(200):
    typeR = random.choice(['Lab', 'Lecture', 'Lecture', 'Lounge', 'Meeting', 'Multi-purpose', 'Office', 'Resource'])
    equipment = random.choice(
        ['3D Printer', 'Chairs, Tables, Boards', 'Chemistry', 'Computers', 'Conference', 'Engineering', 'Lecture',
         'Library', 'Maker-space', 'Physics'])
    building = random.choice(['IST', 'BARC'])
    # constraints
    if typeR == 'Lab' and (equipment == 'Chemistry' or equipment == 'Engineering'):
        accessibility = 'Lab Safety'
    else:
        accessibility = random.choice(['Braille', 'Chairs', 'Lighting', 'Tables', 'Mobility', 'None'])

    if typeR == 'Office':
        if building == 'IST':
            floor = 2
        else:
            floor = random.randint(1, 2)
        schedule = 'Office Hours'
    elif typeR == 'Lecture':
        floor = 1
    else:
        floor = random.randint(1, 2)

    if building == 'IST' and typeR != 'Office':
        schedule = '6:15am - 12am'
    elif building == 'BARC' and typeR != 'Office':
        schedule = '6:15am - 12am with 7pm quiet hr'


    row = {
        'Room ID': random.randint(1000, 5000),
        'Building': building,
        'Capacity': random.randint(10, 52),
        'Type': typeR,
        'Equipment': equipment,
        'Software': random.choice(['General Windows OS', 'Software Oriented', 'None']),
        'Accessibility': accessibility,
        'Floor': floor,
        'Access': random.choice(['Staff', 'Students', 'All']),
        'Schedule': schedule,
        'Requestable': random.choice(['Yes', 'No'])
    }
    roomData.append(row)

df = pd.DataFrame(roomData)

df.to_csv("rooms.csv", index=False, encoding='UTF-8')
"""

rooms_df = pd.read_csv("rooms.csv")

"""
# user data
fake = Faker()
userData = []

for i in range(1, 10001):
    user_id = str(i).zfill(4)

    preferences = ['Building', 'Capacity', 'Type', 'Equipment', 'Software', 'Accessibility', 'Floor', 'Access', 'Schedule', 'Requestable']
    pref5, pref4, pref3, pref2, pref1 = random.sample(preferences, 5)

    ratings = [1, 2, 3, 4, 5]

    row = {
        'User ID': user_id,
        'Pref 1': pref1,
        'Rating 1': ratings[0],
        'Pref 2': pref2,
        'Rating 2': ratings[1],
        'Pref 3': pref3,
        'Rating 3': ratings[2],
        'Pref 4': pref4,
        'Rating 4': ratings[3],
        'Pref 5': pref5,
        'Rating 5': ratings[4]
    }
    userData.append(row)

user_df = pd.DataFrame(userData)

user_df.to_csv("users.csv", index=False, encoding='UTF-8')
"""
user_df = pd.read_csv("users.csv", dtype=str)  # Make sure values are strings
user_df.replace({'': np.nan, ' ': np.nan}, inplace=True)  # Convert blank spaces to nan
print(user_df.head())

# Concatenate preferences into 1 DataFrame with uniform column naming
popular_preferences = pd.concat([
    user_df[['User ID', 'Pref 5', 'Rating 5']].rename(columns={'Pref 5': 'Preference', 'Rating 5': 'Rating'}),
    user_df[['User ID', 'Pref 4', 'Rating 4']].rename(columns={'Pref 4': 'Preference', 'Rating 4': 'Rating'}),
    user_df[['User ID', 'Pref 3', 'Rating 3']].rename(columns={'Pref 3': 'Preference', 'Rating 3': 'Rating'}),
    user_df[['User ID', 'Pref 2', 'Rating 2']].rename(columns={'Pref 2': 'Preference', 'Rating 2': 'Rating'}),
    user_df[['User ID', 'Pref 1', 'Rating 1']].rename(columns={'Pref 1': 'Preference', 'Rating 1': 'Rating'}),
])

popular_preferences = popular_preferences.dropna(subset=['Preference'])

# Calculate by sorting the average rating for each preference
average_ratings = popular_preferences.groupby('Preference')['Rating'].mean().sort_values(ascending=False)
# Get top preferences
top_preferences = average_ratings.head(5).index.tolist()

print("\nTop 5 recommended Preferences based on past users:")
for i in range(5, 0, -1):
    print(f'{i}. {top_preferences[i - 1]}')

def calculate_rating_distance(user_rating, average_rating):
    # Find distance between user's rating and average rating
    return abs(user_rating - average_rating)  # Make absolute to avoid negatives


def recommend_last_preference(user_df, popular_preferences):
    user_df['Recommended Pref 5'] = np.nan  # Ensure there's a column for the recommended preference

    preference_avg_ratings = popular_preferences.groupby('Preference')['Rating'].mean()
    preference_popularity = popular_preferences['Preference'].value_counts()

    # Loop through users
    for user_index, user_row in user_df.iterrows():
        if pd.isna(user_row['Pref 5']):
            user_prefs = [(user_row['Pref 1'], user_row['Rating 1']), (user_row['Pref 2'], user_row['Rating 2']), (user_row['Pref 3'], user_row['Rating 3']), (user_row['Pref 4'], user_row['Rating 4'])]
            user_prefs = [pref for pref in user_prefs if pd.notna(pref[0])]

            print(f"\nUser {user_row['User ID']}'s current preferences and ratings: {user_prefs}")

            scores = []
            for preference, avg_rating in preference_avg_ratings.items():
                if preference in [pref[0] for pref in user_prefs]:
                    continue  # Skip already chosen preferences

                distances = [calculate_rating_distance(float(user_pref[1]), avg_rating) for user_pref in user_prefs]
                avg_distance = sum(distances) / len(distances)
                score = (1 / (avg_distance + 1)) * preference_popularity.get(preference, 0)
                scores.append((preference, score))

            if scores:
                scores.sort(key=lambda x: x[1], reverse=True)  # Sort scores to find the highest one
                recommended_pref = scores[0][0]
                user_df.at[user_index, 'Recommended Pref 5'] = recommended_pref
                print(f"Recommended for user {user_row['User ID']}: {recommended_pref} with score: {scores[0][1]}")
            else:
                print(f"No recommendation found for user {user_row['User ID']}")

    return user_df

# Make sure to convert 'Rating' to float before using it in calculations
popular_preferences['Rating'] = popular_preferences['Rating'].astype(float)
new_user_df = recommend_last_preference(user_df, popular_preferences)
print("\n")

print(new_user_df[['User ID', 'Pref 1', 'Pref 2', 'Pref 3', 'Pref 4', 'Pref 5', 'Recommended Pref 5']].head())
# """
