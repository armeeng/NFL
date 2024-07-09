import requests
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime, timezone
import time
from dotenv import load_dotenv
import os

# Start the timer
start_time = time.time()

# Define the API keys and endpoints
WEATHER_API_KEY = os.getenv('API_KEY')
BASE_GAME_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary?event="
FIRST_API_URL_TEMPLATE = "http://sports.core.api.espn.com/v2/sports/football/leagues/nfl/events/{}/competitions/{}/competitors/{}/roster"
SECOND_API_URL_TEMPLATE = "http://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{}/roster"


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
WHERE has_inputs = FALSE
"""

# Load data into a DataFrame
notSeen_finished_noInputs = pd.read_sql(query, con=engine)

# Function to get teams info
def get_teams_info(game_id):
    game_summary_url = BASE_GAME_URL + str(game_id)
    response = requests.get(game_summary_url)
    game_data = response.json()
    home_team_info = game_data["boxscore"]["teams"][1]["team"]
    away_team_info = game_data["boxscore"]["teams"][0]["team"]
    return home_team_info, away_team_info

# Function to get game date
def get_game_date(game_id):
    url = BASE_GAME_URL + str(game_id)
    response = requests.get(url)
    game_info = response.json()
    game_date_str = game_info['header']['competitions'][0]['date']
    game_date = datetime.strptime(game_date_str, '%Y-%m-%dT%H:%MZ').replace(tzinfo=timezone.utc)
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    days_since_epoch = (game_date - epoch).days
    return {
        'game_id': game_id, 
        'days_since_epoch': days_since_epoch, 
        'hour': game_date.hour, 
        'minute': game_date.minute,
        'day': game_date.day,
        'month': game_date.month,
        'year': game_date.year
    }

# Function to get data from the first API
def get_data_from_first_api(url):
    response = requests.get(url)
    if response.status_code == 404:
        return None
    return response.json()

# Function to get data from the second API
def get_data_from_second_api(url):
    response = requests.get(url)
    if response.status_code != 200:
        return None
    return response.json()

# Function to extract athlete details
def get_athlete_details(url, event_id, team_id):
    response = requests.get(url)
    athlete_data = response.json()
    college_id = athlete_data.get("college", {}).get("$ref", "").split("/")[-1].split("?")[0] or 0
    player_name = athlete_data.get("displayName", "")
    return {
        "player_name": player_name,
        "weight": athlete_data.get("weight", 0),
        "height": athlete_data.get("height", 0),
        "date_of_birth": convert_date_of_birth(athlete_data.get("dateOfBirth", "")),
        "position_id": int(athlete_data.get("position", {}).get("id", 0)),
        "college_id": int(college_id),
        "game_id": int(event_id),
        "team_id": int(team_id)
    }

# Function to convert date of birth to days since Unix epoch
def convert_date_of_birth(date_str):
    if not date_str:
        return 0
    dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    return (dt - epoch).days

# Function to process player data from the first API
def process_player_data_first_api(player_entry, event_id, team_id):
    athlete_url = player_entry["athlete"]["$ref"]
    athlete_details = get_athlete_details(athlete_url, event_id, team_id)
    return {
        "player_id": int(player_entry["playerId"]),
        "player_name": athlete_details["player_name"],
        "healthy": not player_entry["didNotPlay"],
        **athlete_details
    }

# Function to process player data from the second API
def process_player_data_second_api(player, event_id, team_id):
    healthy = not player.get("injuries")
    college_id = player.get("college", {}).get("id", 0)
    return {
        "player_id": int(player["id"]),
        "player_name": player["fullName"],
        "healthy": healthy,
        "weight": player.get("weight", 0),
        "height": player.get("height", 0),
        "date_of_birth": convert_date_of_birth(player.get("dateOfBirth", "")),
        "position_id": int(player["position"]["id"]),
        "college_id": int(college_id),
        "game_id": int(event_id),
        "team_id": int(team_id)
    }

# Function to get venue information
def get_venue_info(game_id):
    url = BASE_GAME_URL + str(game_id)
    response = requests.get(url)
    data = response.json()
    venue_id = data["gameInfo"]["venue"]["id"]
    venue_url = f"https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/venues/{venue_id}?lang=en&region=us"
    venue_response = requests.get(venue_url)
    venue_data = venue_response.json()
    return {
        "game_id": game_id,
        "venue_id": venue_id,
        "grass": venue_data["grass"],
        "indoor": venue_data["indoor"],
        "zip_code": venue_data["address"].get("zipCode")
    }

# Function to get weather information
def get_weather(game_id, year, month, day, zip_code):
    date = f"{year}-{month:02d}-{day:02d}"
    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{zip_code}/{date}/{date}?unitGroup=metric&key={WEATHER_API_KEY}&contentType=json"
    response = requests.get(url)
    if response.status_code == 200:
        day_data = response.json()['days'][0]
        day_data = {k: (v if v is not None else 0) for k, v in day_data.items()}
        return {
            "game_id": game_id,
            "temp_max": day_data.get('tempmax', 0),
            "temp_min": day_data.get('tempmin', 0),
            "temp": day_data.get('temp', 0),
            "feels_like_max": day_data.get('feelslikemax', 0),
            "feels_like_min": day_data.get('feelslikemin', 0),
            "feels_like": day_data.get('feelslike', 0),
            "dew": day_data.get('dew', 0),
            "humidity": day_data.get('humidity', 0),
            "precipitation": day_data.get('precip', 0),
            "precipitation_probability": day_data.get('precipprob', 0),
            "precipitation_coverage": day_data.get('precipcover', 0),
            "snow": day_data.get('snow', 0),
            "snow_depth": day_data.get('snowdepth', 0),
            "wind_gust": day_data.get('windgust', 0),
            "wind_speed": day_data.get('windspeed', 0),
            "wind_direction": day_data.get('winddir', 0),
            "pressure": day_data.get('pressure', 0),
            "cloud_cover": day_data.get('cloudcover', 0),
            "visibility": day_data.get('visibility', 0),
            "solar_radiation": day_data.get('solarradiation', 0),
            "solar_energy": day_data.get('solarenergy', 0),
            "uv_index": day_data.get('uvindex', 0),
            "severe_risk": day_data.get('severerisk', 0),
            "sunrise_epoch": day_data.get('sunriseEpoch', 0),
            "sunset_epoch": day_data.get('sunsetEpoch', 0),
            "moon_phase": day_data.get('moonphase', 0)
        }
    else:
        return None

# Iterate over the rows in the DataFrame
for index, row in notSeen_finished_noInputs.iterrows():
    game_id = row['game_id']
    print(game_id)
    
    # Get home team and away team information
    home_team_info, away_team_info = get_teams_info(game_id)
    home_team_id = home_team_info["id"]
    home_team_name = home_team_info["abbreviation"]
    away_team_id = away_team_info["id"]
    away_team_name = away_team_info["abbreviation"]
    team_data = {
        'game_id': game_id,
        'home_id': home_team_id,
        'home_name': home_team_name,
        'away_id': away_team_id,
        'away_name': away_team_name
    }
    
    # Get game date information
    game_date = get_game_date(game_id)
    
    # Get players information
    players_data = []
    for team_id in [home_team_id, away_team_id]:
        first_api_url = FIRST_API_URL_TEMPLATE.format(game_id, game_id, team_id)
        second_api_url = SECOND_API_URL_TEMPLATE.format(team_id)
        data = get_data_from_first_api(first_api_url)
        if data:
            players = [process_player_data_first_api(entry, game_id, team_id) for entry in data["entries"]]
        else:
            data = get_data_from_second_api(second_api_url)
            if data:
                players = [process_player_data_second_api(player, game_id, team_id) for section in data["athletes"] for player in section["items"]]
            else:
                players = []
        players_data.extend(players)
    
    # Get venue information
    venue_info = get_venue_info(game_id)
    
    # Get weather information
    year, month, day = game_date['year'], game_date['month'], game_date['day']
    zip_code = venue_info['zip_code']
    weather_info = get_weather(game_id, year, month, day, zip_code)

    # Check if all required data is present
    if team_data and game_date and players_data and venue_info and weather_info:
        # Send data to SQL
        pd.DataFrame([team_data]).to_sql('Teams', con=engine, if_exists='append', index=False)
        pd.DataFrame([game_date]).to_sql('Dates', con=engine, if_exists='append', index=False)
        pd.DataFrame(players_data).to_sql('Players', con=engine, if_exists='append', index=False)
        pd.DataFrame([venue_info]).to_sql('Venues', con=engine, if_exists='append', index=False)
        pd.DataFrame([weather_info]).to_sql('Weather', con=engine, if_exists='append', index=False)
        
        # Update the 'has_inputs' column in the 'Games' table
        with engine.connect() as connection:
            transaction = connection.begin()
            try:
                update_query = text("""
                UPDATE Games
                SET has_inputs = TRUE
                WHERE game_id = :game_id
                """)
                connection.execute(update_query, {'game_id': game_id})
                transaction.commit()
                print(f"Update has_inputs to TRUE for game id: {game_id}")
            except Exception as e:
                transaction.rollback()
                print(f"Transaction failed: {e}")

# Calculate the elapsed time
end_time = time.time()
elapsed_time = end_time - start_time

# Print the elapsed time
print(f"Script execution time: {elapsed_time:.2f} seconds")
print("Update completed successfully.")
