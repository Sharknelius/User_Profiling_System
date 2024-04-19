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
# Second implementation
def calculate_similarity_score(user_prefs, other_user_prefs, total_preferences=10):
    """
    Calculate the similarity score between two users based on their preferences.
    The score is the inverse of the average squared difference in ratings for shared preferences.
    Unrated preferences are counted as 0. Lower scores indicate more similarity.
    """
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
            sum_squared_diff += (1 - rating) ** 2
            compared_preferences += 1

    # Calculate average squared difference
    avg_squared_diff = sum_squared_diff / max(compared_preferences, total_preferences)

    return avg_squared_diff

def recommend_last_preference2(user_df, popular_preferences):
    # Ensure there's a column for the recommended fifth preference
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

    # Iterate through all users
    for user_id, user_prefs in users_preferences.items():
        # Calculate similarity with every other user
        similarity_scores = []
        for other_user_id, other_user_prefs in users_preferences.items():
            if user_id != other_user_id:
                similarity_score = calculate_similarity_score(user_prefs, other_user_prefs)
                if similarity_score <= 0.4:  # Only consider scores of 0.4 or lower
                    similarity_scores.append((other_user_id, similarity_score))

        # Check if there are any suitable similarity scores to consider
        if not similarity_scores:
            print(f"No suitable similar users found for user {user_id} with a score of 0.2 or below.")
        else:
            # Sort the similarity scores in ascending order (lower score = more similar)
            similarity_scores.sort(key=lambda x: x[1])
            # Select the three most similar users based on the lowest similarity scores
            similar_users = similarity_scores[:3]

            # Display information about the two most similar users
            print(f"Top similar users for user {user_id} with scores ≤ 0.4 are:")
            for sim_user_id, sim_score in similar_users:
                print(f"User ID: {sim_user_id} with similarity score: {sim_score}")

            # Proceed with recommendation logic based on the most similar user
            if similar_users:
                most_similar_user_id = similar_users[0][0]
                most_similar_user_prefs = users_preferences[most_similar_user_id]

                recommended_pref_found = False
                for pref in most_similar_user_prefs.keys():
                    if pref not in user_prefs:
                        recommended_pref = popular_preferences[popular_preferences['Preference'] == pref]['Preference'].values[0]
                        user_df.loc[user_df['User ID'] == user_id, 'Recommended Pref 5'] = recommended_pref
                        recommended_pref_found = True
                        print(f"Recommended for user {user_id}: {recommended_pref}")
                        break

                if not recommended_pref_found:
                    print(f"Could not find a new preference to recommend to user {user_id} based on similar users.")

    return user_df

popular_preferences1['Rating'] = popular_preferences1['Rating'].astype(float)

new_user_df2 = recommend_last_preference2(user_df, popular_preferences1)
print("\n")
print(new_user_df2[['User ID', 'Pref 1', 'Pref 2', 'Pref 3', 'Pref 4', 'Pref 5', 'Recommended Pref 5']].head(21))
