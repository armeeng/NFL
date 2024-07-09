import requests
from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

# Set display options to show all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Define the base URL for the ESPN API
base_url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"

# Define the starting and ending years
start_year = 2009
end_year = 2023

# Initialize an empty list to store game details
all_games = []

# Get current date
current_date = datetime.now()

# Loop through each year
for year in range(start_year, end_year + 1):
    # Loop through each seasontype (regular season and playoffs)
    for seasontype in [2, 3]:  # Regular season (2), Post Season (3)
        # Loop through each week
        for week_num in range(1, 19):  # Regular season up to week 18, playoffs handled separately
            # Skip week 4 of post season (pro bowl)
            if seasontype == 3 and week_num == 4:
                continue
            
            # Define the request URL with parameters
            url = f"{base_url}?dates={year}&seasontype={seasontype}&week={week_num}"

            print(f"Year: {year}, Week: {week_num}, Season Type: {seasontype}")

            # Send GET request to the API
            response = requests.get(url)

            # Extract game details from the response JSON
            data = response.json()
            for event in data['events']:
                game_id = event['id']
                all_games.append({
                    'game_id': game_id,
                    'season_type': seasontype,
                    'week_num': week_num
                })

# Convert list of game details to a pandas DataFrame
df = pd.DataFrame(all_games)
    

df['neural_network_seen'] = False
df['game_finished'] = False
df['has_inputs'] = False
df['has_pbp'] = False
df['has_pred'] = False

load_dotenv()

# Define MySQL database connection details
db_username = os.getenv('DB_USERNAME')
db_password = os.getenv('DB_PASSWORD')
db_host = os.getenv('DB_HOST')
db_port = os.getenv('DB_PORT')
db_name = os.getenv('DB_NAME')

# SQLAlchemy Connection String
connection_str = f"mysql+pymysql://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}"

# Create SQLAlchemy Engine
engine = create_engine(connection_str)

# Send the DataFrame to SQL
df.to_sql('Games', con=engine, if_exists='append', index=False)

print("Game IDs inserted into the table successfully.")
