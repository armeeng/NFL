import requests
import pandas as pd
from sqlalchemy import create_engine, text
import re
from dotenv import load_dotenv
import os

def is_game_finished(game_id):
    url = f"https://cdn.espn.com/core/nfl/game?xhr=1&gameId={game_id}"
    response = requests.get(url)
    pattern = re.compile(r'"status":\s*{[^}]*"id":\s*"3"')
    match = re.search(pattern, response.text)
    print(game_id)
    return bool(match)

# Set display options to show all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

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
SELECT game_id FROM Games
WHERE game_finished = FALSE
"""

# Load data into a DataFrame
not_finished = pd.read_sql(query, con=engine)
not_finished['game_finished'] = not_finished['game_id'].apply(is_game_finished)

# Log the data to be updated
print(not_finished)

# Update the game_finished column in the Games table in SQL
with engine.connect() as connection:
    transaction = connection.begin()
    try:
        for index, row in not_finished.iterrows():
            update_query = text("""
            UPDATE Games
            SET game_finished = :game_finished
            WHERE game_id = :game_id
            """)
            connection.execute(update_query, {'game_finished': row['game_finished'], 'game_id': row['game_id']})
            print(f"Updated game_id {row['game_id']} to game_finished {row['game_finished']}")
        transaction.commit()
    except Exception as e:
        transaction.rollback()
        print(f"Transaction failed: {e}")
