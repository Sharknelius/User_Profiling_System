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
user_df = pd.read_csv("users.csv")
print(user_df.head())

# Concatenate preferences into 1 DataFrame with uniform column naming
popular_preferences = pd.concat([
    user_df[['User ID', 'Pref 1', 'Pref 1 Rating']].rename(columns={'Pref 1': 'Preference', 'Pref 1 Rating': 'Rating'}),
    user_df[['User ID', 'Pref 2', 'Pref 2 Rating']].rename(columns={'Pref 2': 'Preference', 'Pref 2 Rating': 'Rating'}),
    user_df[['User ID', 'Pref 3', 'Pref 3 Rating']].rename(columns={'Pref 3': 'Preference', 'Pref 3 Rating': 'Rating'}),
])

# Calculate by sorting the average rating for each preference
average_ratings = popular_preferences.groupby('Preference')['Rating'].mean().sort_values(ascending=False)
 # row based similarity recommendation system
# Get top preferences
top_preferences = average_ratings.head(5).index.tolist()

print("\nTop 5 recommended Preferences based on past users:")
for i in range(5):
    print(f'{i + 1}. {top_preferences[i]}')
