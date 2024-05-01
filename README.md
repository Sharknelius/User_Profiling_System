# User Profiling System
## Three Folders:
Each folder contains a main.py and two csv files.
1. Dataset_built
- Run main.py
- Contains code for creating the rooms.csv and users.csv
- The csv files already in the folder are examples of how the csvs will look
- The only necessary file to have is the main.py
2. Recommendation1
- Run main.py with users.csv and rooms.csv in the same directory
- Reads the user.csv and rooms.csv (they are necessary this time)
- Will calculate the popularity of preferences
- Then calculates the recommended fifth preference for users with one null preference
- Uses averages of each preference to find the new preference
3. Recommendation2
- Similar to Recommendation1, but calculating new preference is different
- Compares all users to one another to find similarity scores

