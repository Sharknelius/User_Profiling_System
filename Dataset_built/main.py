import random
import pandas as pd
import numpy as np
from faker import Faker

# Set pandas table display settings
pd.set_option('display.max_columns', None)  # all columns

pd.set_option('display.width', 1000)

pd.set_option('display.max_rows', 20000)

# Using the faker library, a dataset can be quickly created
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

rooms_df = pd.read_csv("rooms.csv")
print(rooms_df.head(21))

# user data
fake = Faker()
userData = []

for i in range(1, 10001):
    user_id = str(i).zfill(4)
    
    # Possible preferences
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

user_df = pd.read_csv("users.csv", dtype=str, header=0)  # Make sure values are strings
user_df.replace({'': np.nan, ' ': np.nan}, inplace=True)  # Convert blank spaces to nan
print(user_df.head(21))
