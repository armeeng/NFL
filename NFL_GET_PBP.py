import requests
import pandas as pd
from sqlalchemy import create_engine, text
import re
from fuzzywuzzy import process, fuzz
from dotenv import load_dotenv
import os

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

def extract_team_id(team_ref):
    return team_ref.split('/')[-1].split('?')[0] if team_ref else None

def get_play_by_play_data(event_id):
    url = f"https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/events/{event_id}/competitions/{event_id}/plays?limit=300"
    response = requests.get(url)
    data = response.json()
    
    plays = []
    for item in data['items']:
        play = {
            'game_id': event_id,
            'id': item.get('id'),
            'type': item.get('type', {}).get('text'),
            'text': item.get('text'),
            'awayScore': item.get('awayScore'),
            'homeScore': item.get('homeScore'),
            'period': item.get('period', {}).get('number'),
            'clock': item.get('clock', {}).get('value'),
            'scoringPlay': item.get('scoringPlay'),
            'team_id': extract_team_id(item.get('team', {}).get('$ref')),
            'start_down': item.get('start', {}).get('down'),
            'start_distance': item.get('start', {}).get('distance'),
            'start_yardLine': item.get('start', {}).get('yardLine'),
            'start_yardsToEndzone': item.get('start', {}).get('yardsToEndzone'),
            'start_team': extract_team_id(item.get('start', {}).get('team', {}).get('$ref')),
            'end_down': item.get('end', {}).get('down'),
            'end_distance': item.get('end', {}).get('distance'),
            'end_yardLine': item.get('end', {}).get('yardLine'),
            'end_yardsToEndzone': item.get('end', {}).get('yardsToEndzone'),
            'end_team': extract_team_id(item.get('end', {}).get('team', {}).get('$ref')),
            'statYardage': item.get('statYardage')
        }
        plays.append(play)
    
    df = pd.DataFrame(plays)
    return df

def get_player_data_for_game_id(game_id):
    query = f"""
    SELECT * FROM NFL.Players
    WHERE game_id = '{game_id}'
    """
    return pd.read_sql(query, con=engine)

def replace_player_names_with_ids(play_by_play_data, player_data):
    # Create a dictionary to map full player names to player IDs
    player_name_to_id = {row['player_name']: row['player_id'] for _, row in player_data.iterrows()}
    player_names = list(player_name_to_id.keys())
    
    def replace_names(text):
        # Replace player names with their IDs
        player_ids = []
        for player_name, player_id in player_name_to_id.items():
            if player_name in text:
                text = text.replace(player_name, str(player_id))
                player_ids.append(str(player_id))
        
        # If full names are not found, proceed with abbreviation-based replacement
        pattern = re.compile(r'\b\w+\.\w+(?:-\w+)*\b')
        words = pattern.findall(text)
        for word in words:
            best_match, score = process.extractOne(word, player_names, scorer=fuzz.partial_ratio)
            if score > 80:
                text = text.replace(word, str(player_name_to_id[best_match]))
                player_ids.append(str(player_name_to_id[best_match]))
        
        return ' '.join(player_ids)
    
    play_by_play_data['text'] = play_by_play_data['text'].apply(replace_names)
    
    return play_by_play_data

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

# Query to select rows from the Games table that the neural network has not seen,
# the game is finished, and it does not have inputs
query = """
SELECT * FROM Games
WHERE game_finished = TRUE
AND has_pbp = FALSE
"""

# Load data into a DataFrame
finished_noPBP = pd.read_sql(query, con=engine)

# List to hold all play-by-play data
all_plays = []

# Iterate over each game ID and get play-by-play data
for event_id in finished_noPBP['game_id']:
    print(event_id)
    player_data = get_player_data_for_game_id(event_id)
    play_by_play_data = get_play_by_play_data(event_id)
    if not play_by_play_data.empty:
        play_by_play_data = play_by_play_data.fillna(0)
        play_by_play_data = replace_player_names_with_ids(play_by_play_data, player_data)
        all_plays.append(play_by_play_data)
    else:
        print("NO PLAY BY PLAY DATA AVAILABLE")

# Combine all play-by-play data into a single DataFrame
all_plays_df = pd.concat(all_plays, ignore_index=True)

# Save the combined DataFrame to the database
all_plays_df.to_sql('PlayByPlays', con=engine, if_exists='append', index=False)

# Update the 'has_pbp' column in the 'Games' table for the processed game IDs
processed_game_ids = set(all_plays_df['game_id'])

with engine.connect() as connection:
    transaction = connection.begin()
    try:
        for game_id in processed_game_ids:
            update_query = text("""
            UPDATE Games
            SET has_pbp = TRUE
            WHERE game_id = :game_id
            """)
            connection.execute(update_query, {'game_id': game_id})
            print(f"Updated game_id {game_id} to has_pbp TRUE")
        transaction.commit()
    except Exception as e:
        transaction.rollback()
        print(f"Transaction failed: {e}")

print("Update completed successfully.")
