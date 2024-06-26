import pandas as pd
import numpy as np

# Data table display settings
# Allow for more visible rows and columns
pd.set_option('display.max_columns', None)  # all columns

pd.set_option('display.width', 1000)

pd.set_option('display.max_rows', 20000)

rooms_df = pd.read_csv("rooms.csv")

# Convert CSV to dataframe
user_df = pd.read_csv("users.csv", dtype=str, header=0)  # Make sure values are strings
user_df.replace({'': np.nan, ' ': np.nan}, inplace=True)  # Convert blank spaces to nan
print(user_df.head())

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

    preference_avg_ratings = popular_preferences.groupby('Preference')['Rating'].mean()  # Calculates the average
    # rating for each preference
    preference_popularity = popular_preferences['Preference'].value_counts()  # Finds amount of time each preference
    # is referenced in total

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

            # print(f"\nUser {user_row['User ID']}'s current preferences and ratings: {user_prefs}")

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
                # print(f"Recommended for user {user_row['User ID']}: {recommended_pref} with score: {scores[0][1]}")

    return user_df

popular_preferences1['Rating'] = popular_preferences1['Rating'].astype(float)

new_user_df = recommend_last_preference(user_df, popular_preferences1)
print(new_user_df[['User ID', 'Pref 1', 'Pref 2', 'Pref 3', 'Pref 4', 'Pref 5', 'Recommended Pref 5']].head(21))
print("\n")
