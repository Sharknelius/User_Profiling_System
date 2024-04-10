import csv
import random
import pandas as pd
from pprint import pprint
import numpy as np
# import sklearn
# import seaborn as sns
# import matplotlib.pyplot as plt
# from surprise import Reader, Dataset
# from faker import Faker

rooms_df = pd.read_csv("rooms.csv")

user_df = pd.read_csv("users.csv", dtype=str, header=0)  # Make sure values are strings
user_df.replace({'': np.nan, ' ': np.nan}, inplace=True)  # Convert blank spaces to nan
print(user_df.head())

# Concatenate preferences into 1 DataFrame with uniform column naming
popular_preferences0 = pd.concat([
    user_df[['User ID', 'Pref 1', 'Rating 1']].rename(columns={'Pref 1': 'Preference', 'Rating 1': 'Rating'}),
    user_df[['User ID', 'Pref 2', 'Rating 2']].rename(columns={'Pref 2': 'Preference', 'Rating 2': 'Rating'}),
    user_df[['User ID', 'Pref 3', 'Rating 3']].rename(columns={'Pref 3': 'Preference', 'Rating 3': 'Rating'}),
    user_df[['User ID', 'Pref 4', 'Rating 4']].rename(columns={'Pref 4': 'Preference', 'Rating 4': 'Rating'}),
    user_df[['User ID', 'Pref 5', 'Rating 5']].rename(columns={'Pref 5': 'Preference', 'Rating 5': 'Rating'}),
])

popular_preferences0 = popular_preferences0.dropna(subset=['Preference'])
# Rating is now numeric
popular_preferences0['Rating'] = pd.to_numeric(popular_preferences0['Rating'])

top_prefs_by_rating = {}

# This for loop will check for each rating the most popular preference
for rating in [5, 4, 3, 2, 1]:
    preferences_for_rating = popular_preferences0[popular_preferences0['Rating'] == rating]

    # Count number of preference and sort them
    preference_count = preferences_for_rating['Preference'].value_counts()

    # Get the most popular preference
    if not preference_count.empty:
        top_prefs_by_rating[rating] = preference_count.idxmax()

print("\nMost popular preferences for each rating:")
for i in range(5, 0, -1):
    rating_key = str(i)  # Set rating to string
    # Check if the rating exists
    if i in top_prefs_by_rating:
        print(f'Rating {i}: {top_prefs_by_rating[i]}')

print("\n")

# Weighted score across all preferences
popular_preferences1 = pd.concat([
    user_df[['User ID', 'Pref 1', 'Rating 1']].rename(columns={'Pref 1': 'Preference', 'Rating 1': 'Rating'}),
    user_df[['User ID', 'Pref 2', 'Rating 2']].rename(columns={'Pref 2': 'Preference', 'Rating 2': 'Rating'}),
    user_df[['User ID', 'Pref 3', 'Rating 3']].rename(columns={'Pref 3': 'Preference', 'Rating 3': 'Rating'}),
    user_df[['User ID', 'Pref 4', 'Rating 4']].rename(columns={'Pref 4': 'Preference', 'Rating 4': 'Rating'}),
    user_df[['User ID', 'Pref 5', 'Rating 5']].rename(columns={'Pref 5': 'Preference', 'Rating 5': 'Rating'}),
])

popular_preferences1 = popular_preferences1.dropna(subset=['Preference'])
# Rating is now numeric
popular_preferences1['Rating'] = pd.to_numeric(popular_preferences1['Rating'])

popular_preferences1['WeightedScore'] = popular_preferences1['Rating']

# Group by 'Preference', sum the 'WeightedScore', and sort the results
weighted_scores = popular_preferences1.groupby('Preference')['WeightedScore'].sum().sort_values(ascending=False)

# Get the top 10 preferences based on their weighted scores
top_10_prefs = weighted_scores.head(10)

print("Top 10 preferences based on weighted scores:")
for i, (preference, score) in enumerate(top_10_prefs.items(), start=1):
    print(f"{i}. {preference}'s total weighted score = {score}")

print("\n")

"""does not account for weight of rating
# Calculate by sorting the average rating for each preference
average_ratings = popular_preferences.groupby('Preference')['Rating'].mean().sort_values(ascending=False)
# Get top preferences
top_preferences = average_ratings.head(10).index.tolist()

print("\nPreferences ranked by popularity (5th preference is highest):")
for i in range(1, 11):
    print(f'{i}. {top_preferences[10 - i]}')
"""
def calculate_rating_distance(user_rating, average_rating, weight):
    # Find distance between user's rating and average rating
    return abs(user_rating - average_rating * weight)  # Make absolute to avoid negatives


def recommend_last_preference(user_df, popular_preferences):
    user_df['Recommended Pref 5'] = np.nan  # Ensures there's a column for recommended preference
    user_df = user_df.astype({'Recommended Pref 5': 'object'})

    preference_avg_ratings = popular_preferences.groupby('Preference')['Rating'].mean()  # Calculates average rating for each preference
    preference_popularity = popular_preferences['Preference'].value_counts()  # Finds total count of a preference

    # Each preference is given a weight
    weights = {'Pref 5': 5, 'Pref 4': 4, 'Pref 3': 3, 'Pref 2': 2, 'Pref 1': 1}

    # Loop through users
    for user_index, user_row in user_df.iterrows():
        if pd.isna(user_row['Pref 5']):  # If the user is missing a 5th preference
            user_prefs = [
                (user_row['Pref 1'], float(user_row['Rating 1']), weights['Pref 1']),
                (user_row['Pref 2'], float(user_row['Rating 2']), weights['Pref 2']),
                (user_row['Pref 3'], float(user_row['Rating 3']), weights['Pref 3']),
                (user_row['Pref 4'], float(user_row['Rating 4']), weights['Pref 4']),
            ]
            user_prefs = [pref for pref in user_prefs if pd.notna(pref[0])]

            print(f"\nUser {user_row['User ID']}'s current preferences and ratings: {user_prefs}")

            scores = []
            for preference, avg_rating in preference_avg_ratings.items():
                if preference in [pref[0] for pref in user_prefs]:
                    continue  # Skip already chosen preferences
                # Call calculation rating distance function and input current user pref and avg rating
                # Calculates a score based on how close the user's ratings for their existing
                # preferences are to the average rating of the potential preference and popularity
                weighted_distances = [
                    calculate_rating_distance(user_pref[1], avg_rating, user_pref[2]) for user_pref in user_prefs
                ]
                avg_weighted_distance = sum(weighted_distances) / sum([pref[2] for pref in user_prefs])
                score = (1 / (avg_weighted_distance + 1)) * preference_popularity.get(preference, 0)
                scores.append((preference, score))

            if scores:
                scores.sort(key=lambda x: x[1], reverse=True)  # Sort scores to find the highest one
                recommended_pref = scores[0][0]
                user_df.at[user_index, 'Recommended Pref 5'] = recommended_pref
                print(f"Recommended for user {user_row['User ID']}: {recommended_pref} with score: {scores[0][1]}")
            else:
                print(f"No recommendation found for user {user_row['User ID']}")

    return user_df

# Second implementation
def calculate_similarity_score(user_prefs, other_user_prefs, total_preferences=10):
    # Initialize sum of squared differences and count of compared preferences
    sum_squared_diff = 0
    compared_preferences = 0

    # Iterate through each preference in the first user
    for pref_name, rating in user_prefs.items():
        # Get other user's rating for = preference, default to 0 if unrated
        other_rating = other_user_prefs.get(pref_name, 0)
        sum_squared_diff += (rating - other_rating) ** 2
        compared_preferences += 1

    # Iterate through each preference in the second user
    for pref_name, rating in other_user_prefs.items():
        if pref_name not in user_prefs:
            sum_squared_diff += (0 - rating) ** 2
            compared_preferences += 1

    # Calculate average squared difference, divide by 10 or 9
    avg_squared_diff = sum_squared_diff / max(compared_preferences, total_preferences)

    return avg_squared_diff

def recommend_last_preference2(user_df, popular_preferences):
    # Add a column for the recommended fifth preference if it doesn't exist
    user_df['Recommended Pref 5'] = np.nan
    user_df = user_df.astype({'Recommended Pref 5': 'object'})

    # Convert user preferences and ratings into a more usable format
    users_preferences = {}
    for index, row in user_df.iterrows():
        prefs = {}
        for i in range(1, 6):
            pref_name = row.get(f'Pref {i}')
            rating = row.get(f'Rating {i}')
            if pd.notna(pref_name) and pd.notna(rating):
                prefs[pref_name] = float(rating)
        users_preferences[row['User ID']] = prefs

    # Iterate through users missing a fifth preference
    for user_id, user_prefs in users_preferences.items():
        # Calculate similarity with every other user
        similarity_scores = []
        for other_user_id, other_user_prefs in users_preferences.items():
            # if user_id == other_user_id or len(other_user_prefs) != 4:
            if len(other_user_prefs) != 4:
                continue  # Skip self comparison and users unable to recommend for
            # Return similarity score
            similarity_score = calculate_similarity_score(user_prefs, other_user_prefs)
            # print({similarity_score})
            similarity_scores.append((other_user_id, similarity_score))

        # Check if there are any similarity scores 
        if not similarity_scores:
            # If no similar users are found
            print(f"No similar users found for user {user_id}, unable to recommend a fifth preference.")
        else:
            # Lower score = more similar
            similarity_scores.sort(key=lambda x: x[1])
            # Select the most similar user based on the lowest similarity score
            most_similar_user_id, _ = similarity_scores[0]

            # Fetch the prefs of the most similar user
            most_similar_user_prefs = users_preferences[most_similar_user_id]

            # Try to find a pref that the most similar user has but the current user doesn't
            recommended_pref_found = False
            for pref in most_similar_user_prefs.keys():
                if pref not in user_prefs:
                    # Get ame of pref to recommend
                    recommended_pref = \
                    popular_preferences[popular_preferences['Preference'] == pref]['Preference'].values[0]
                    # Update user_df with recommended preference
                    user_df.loc[user_df['User ID'] == user_id, 'Recommended Pref 5'] = recommended_pref
                    recommended_pref_found = True
                    # print(f"Recommended for user {user_id}: {recommended_pref}")
                    break 

            if not recommended_pref_found:
                # print(f"Could not find a new preference to recommend to user {user_id} based on similar users.")
                continue

    return user_df

popular_preferences1['Rating'] = popular_preferences1['Rating'].astype(float)

print("Distance in average method\n")
new_user_df = recommend_last_preference(user_df, popular_preferences1)
print(new_user_df[['User ID', 'Pref 1', 'Pref 2', 'Pref 3', 'Pref 4', 'Pref 5', 'Recommended Pref 5']].head(20))
print("\n Loading...")

new_user_df2 = recommend_last_preference2(user_df, popular_preferences1)
print("Similarity scores method\n")
print(new_user_df2[['User ID', 'Pref 1', 'Pref 2', 'Pref 3', 'Pref 4', 'Pref 5', 'Recommended Pref 5']].head(20))
