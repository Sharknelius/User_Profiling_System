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
        schedule = '6:15am - 12am with 7pm quiet hours'


    row = {
        'Room ID': random.randint(1000, 5000),
        'Building Name': building,
        'Capacity': random.randint(10, 52),
        'Type': typeR,
        'Equipment': equipment,
        'Software': random.choice(['General Windows OS', 'Software Oriented', 'None']),
        'Accessibility': accessibility,
        'Floor': floor,
        'Access Type': random.choice(['Staff', 'Students', 'All']),
        'Schedule': schedule,
        'Requestable': random.choice(['Yes', 'No'])
    }
    roomData.append(row)

df = pd.DataFrame(roomData)

df.to_csv("rooms.csv", index=False, encoding='UTF-8')
"""

rooms_df = pd.read_csv("rooms.csv")
print(rooms_df.head())
"""
# user data
fake = Faker()
userData = []

for i in range(1, 10001):
    user_id = str(i).zfill(4)

    preferences = ['Capacity', 'Type', 'Equipment', 'Software', 'Accessibility', 'Access Type', 'Schedule', 'Requestable']
    pref1, pref2, pref3 = random.sample(preferences, 3)

    ratings = [3, 2, 1]

    row = {
        'User ID': user_id,
        'Pref 1': pref1,
        'Pref 1 Rating': ratings[0],
        'Pref 2': pref2,
        'Pref 2 Rating': ratings[1],
        'Pref 3': pref3,
        'Pref 3 Rating': ratings[2]
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


def find_users_missing_third_pref(user_df):
    # Identify users with exactly 2 non-null preferences
    mask = user_df[['Pref 1', 'Pref 2', 'Pref 3']].notna().sum(axis=1) == 2
    return user_df[mask].index


def recommend_based_on_popularity_and_similarity(user_df, popular_preferences):
    # Initialize a column for recommendations if it doesn't exist
    if 'Recommended Pref 3' not in user_df.columns:
        user_df['Recommended Pref 3'] = np.nan

    for user_index, user_row in user_df.iterrows():
        if pd.isna(user_row['Pref 3']):
            user_prefs = [user_row['Pref 1'], user_row['Pref 2']]
            # Filter out NaN preferences
            user_prefs = [pref for pref in user_prefs if pd.notna(pref)]

            # Find similar users' preferences (excluding the current user's preferences)
            similar_prefs = popular_preferences[~popular_preferences['Preference'].isin(user_prefs)]

            # Aggregate these preferences and count occurrences
            preference_counts = similar_prefs['Preference'].value_counts()

            # Optionally, consider the average rating of these preferences
            preference_ratings = popular_preferences.groupby('Preference')['Rating'].mean()

            # Combine counts with ratings to prioritize both popularity and rating
            combined_scores = preference_counts.combine(preference_ratings, np.multiply, fill_value=0)

            # Recommend the highest scoring preference not already chosen by the user
            for pref in combined_scores.sort_values(ascending=False).index:
                if pref not in user_prefs:
                    user_df.at[user_index, 'New Pref 3'] = pref
                    break

    return user_df

new_user_df = recommend_based_on_popularity_and_similarity(user_df, popular_preferences)
print(new_user_df[['User ID', 'Pref 1', 'Pref 2', 'Pref 3', 'New Pref 3']].head())
