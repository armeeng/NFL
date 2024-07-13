import pandas as pd
from sqlalchemy import create_engine, text
import random
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import time
from dotenv import load_dotenv
import os

start_time = time.time()

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

def get_data_for_game_id(game_id, table_name):
    """
    Retrieves data for a specific game ID from a database table.

    Parameters:
    - game_id: The ID of the game to retrieve data for.
    - table_name: The name of the database table to query.

    Returns:
    A pandas DataFrame containing the data for the specified game ID from the provided table.
    """
    
    query = f"""
    SELECT * FROM {table_name}
    WHERE game_id = '{game_id}'
    """
    return pd.read_sql(query, con=engine)

def get_all_data(table_name):
    """
    Retrieves all data from a specified database table.

    Parameters:
    - table_name: The name of the database table to retrieve data from.

    Returns:
    A pandas DataFrame containing all the data from the specified table.
    """
    query = f"""
    SELECT * FROM {table_name}
    """
    return pd.read_sql(query, con=engine)

def calculate_weather_score(current_weather, historical_weather, weather_weights, weather_thresholds):
    """
    Calculates the weather score based on the current and historical weather data.

    Parameters:
    - current_weather: dictionary containing the current weather data
    - historical_weather: dictionary containing the historical weather data
    - weather_weights: dictionary containing the weights for different weather parameters
    - weather_thresholds: dictionary containing the thresholds for each weather parameter

    Returns:
    - score: the calculated weather score based on the provided data
    """
    score = 0
    for key, threshold in weather_thresholds.items():
        if abs(current_weather[key] - historical_weather[key]) <= threshold:
            score += weather_weights[key]
    return score

def calculate_play_score(play, previous_play, play_weights, play_thresholds):
    """
    Calculates the play score based on the input play data, previous play data, play weights, and play thresholds.

    Parameters:
    - play: dictionary containing the data of the current play
    - previous_play: dictionary containing the data of the previous play
    - play_weights: dictionary containing the weights for different play parameters
    - play_thresholds: dictionary containing the thresholds for each play parameter

    Returns:
    - score: the calculated play score based on the provided data
    """
    score = 0
    for key, threshold in play_thresholds.items():
        if key == 'type':
            if play[key] == previous_play[key]:
                score += play_weights[key]
        elif abs(play[key] - previous_play[key]) <= threshold:
            score += play_weights[key]
    return score

def calculate_games_score(current_game, historical_game, games_weights, games_thresholds):
    """
    Calculates the game score based on the current game data, historical game data, game weights, and game thresholds.

    Parameters:
    - current_game: dictionary containing the data of the current game
    - historical_game: dictionary containing the data of the historical game
    - games_weights: dictionary containing the weights for different game parameters
    - games_thresholds: dictionary containing the thresholds for each game parameter

    Returns:
    - score: the calculated game score based on the provided data
    """
    score = 0
    for key, threshold in games_thresholds.items():
        if abs(current_game[key] - historical_game[key]) <= threshold:
            score += games_weights[key]

    return score

def calculate_date_score(current_datetime, historical_datetime, date_weights, date_thresholds):
    """
    Calculate the score based on the date similarity between the current datetime and historical datetime.
    
    Parameters:
    - current_datetime: The current datetime to compare.
    - historical_datetime: The historical datetime to compare.
    - date_weights: A dictionary with weights for different date components.
    - date_thresholds: A dictionary with thresholds for different date comparisons.
    
    Returns:
    - score: The calculated score based on date similarity.
    """
    score = 0
    
    # Extract time components for the one-hour check
    current_time = current_datetime.time()
    historical_time = historical_datetime.time()
    
    # Calculate the difference in seconds between the two times
    time_difference = abs((datetime.combine(datetime.min, current_time) - datetime.combine(datetime.min, historical_time)).total_seconds())
    
    # Add to score if within one hour (3600 seconds)
    if time_difference <= date_thresholds['one_hour']:
        score += date_weights['one_hour']
    
    # Add to score if within the last 8 months
    if current_datetime - timedelta(days=date_thresholds['eight_months']) <= historical_datetime <= current_datetime:
        score += date_weights['eight_months']
    
    # Add to score if the same day of the week
    if current_datetime.weekday() == historical_datetime.weekday():
        score += date_weights['same_weekday']
    
    return score

def calculate_teams_score(current_team, historical_team, team_weights, team_thresholds):
    """
    Calculate the score based on the similarity between the current team and historical team.

    Parameters:
    - current_team (dict): A dictionary containing the current team's information, including 'home_id' and 'away_id'.
    - historical_team (dict): A dictionary containing the historical team's information, including 'home_id' and 'away_id'.
    - team_weights (dict): A dictionary containing the weights for different team parameters.
    - team_thresholds (dict): A dictionary containing the thresholds for each team parameter.

    Returns:
    - score (int): The calculated score based on the similarity between the current team and historical team.
    """
    score = 0

    current_home_id = current_team['home_id']
    current_away_id = current_team['away_id']
    historical_home_id = historical_team['home_id']
    historical_away_id = historical_team['away_id']

    if current_home_id == historical_home_id:
        score += team_weights['home_id']
    if current_home_id == historical_away_id:
        score += team_weights['home_id_is_away']
    if current_away_id == historical_away_id:
        score += team_weights['away_id']
    if current_away_id == historical_home_id:
        score += team_weights['away_id_is_home']

    return score

def calculate_venues_score(current_venue, historical_venue, venue_weights, venue_thresholds):
    """
    A function that calculates the venue score based on the current and historical venue data.

    Parameters:
    - current_venue: dictionary containing the current venue data
    - historical_venue: dictionary containing the historical venue data
    - venue_weights: dictionary containing the weights for different venue parameters
    - venue_thresholds: dictionary containing the thresholds for each venue parameter

    Returns:
    - score: the calculated venue score based on the provided data
    """
    score = 0
    for key, threshold in venue_thresholds.items():
        if current_venue[key] == historical_venue[key]:
            score += venue_weights[key]

    return score 

def score_games_by_weather(current_weather, all_data_weather, weather_weights, weather_thresholds):
    """
    Calculates the scores for games based on weather conditions by comparing the current weather with historical weather data.

    Parameters:
    - current_weather: dictionary containing the current weather data
    - all_data_weather: DataFrame containing historical weather data
    - weather_weights: dictionary containing weights for weather parameters
    - weather_thresholds: dictionary containing thresholds for weather parameters

    Returns:
    - weather_scores: a dictionary mapping game IDs to the calculated scores based on weather comparison
    """
    weather_scores = {}
    for index, historical_weather in all_data_weather.iterrows():
        game_id = historical_weather['game_id']
        score = calculate_weather_score(current_weather, historical_weather, weather_weights, weather_thresholds)
        weather_scores[game_id] = score
    
    return weather_scores

def score_plays(previous_play, all_data_play_by_plays, play_weights, play_thresholds):
    """
    Calculates the scores of plays based on the provided previous play, all historical plays, play weights, and play thresholds.

    Parameters:
    - previous_play: The previous play information used as a reference for calculating scores.
    - all_data_play_by_plays: A DataFrame containing all historical plays data.
    - play_weights: The weights used in the calculation of play scores.
    - play_thresholds: The thresholds applied in determining the play scores.

    Returns:
    A dictionary containing the scores of each play based on the provided inputs.
    """
    play_scores = {}
    #i = 0
    for index, play in all_data_play_by_plays.iterrows():
        #print(i)
        #i = i + 1
        score = calculate_play_score(play, previous_play, play_weights, play_thresholds)
        play_scores[index] = (play['game_id'], score)  # Store game_id along with score
    
    return play_scores

def score_games_by_date(current_date, all_data_dates, date_weights, date_thresholds):
    """
    Calculate the scores for games based on the dates and historical data.

    Parameters:
    - current_date: dictionary containing the current date information.
    - all_data_dates: DataFrame with historical date information.
    - date_weights: dictionary with weights for different date components.
    - date_thresholds: dictionary with thresholds for different date comparisons.

    Returns:
    - date_scores: dictionary containing the calculated scores for each game based on dates.
    """
    date_scores = {}
    current_datetime = datetime(current_date['year'], current_date['month'], current_date['day'], current_date['hour'], current_date['minute'])
    
    for index, row in all_data_dates.iterrows():
        historical_datetime = datetime(row['year'], row['month'], row['day'], row['hour'], row['minute'])
        game_id = row['game_id']
        
        # Calculate date score
        score = calculate_date_score(current_datetime, historical_datetime, date_weights, date_thresholds)
        
        date_scores[game_id] = score
    
    return date_scores

def score_games_by_games(current_game, all_data_games, games_weights, games_thresholds):
    """
    Calculate the scores for games based on the games data and historical games data.

    Parameters:
    - current_game: dictionary containing the data of the current game
    - all_data_games: DataFrame with historical games data
    - games_weights: dictionary containing weights for different game parameters
    - games_thresholds: dictionary containing thresholds for each game parameter

    Returns:
    - games_scores: dictionary containing the calculated scores for each game based on the games data
    """
    games_scores = {}
    for index, hisoritcal_game in all_data_games.iterrows():
        game_id = hisoritcal_game['game_id']
        score = calculate_games_score(current_game, hisoritcal_game, games_weights, games_thresholds)
        games_scores[game_id] = score
    
    return games_scores

def score_games_by_teams(current_teams, all_data_teams, teams_weights, teams_thresholds):
    """
    Calculates the scores for games based on teams by comparing the current teams with historical team data.

    Parameters:
    - current_teams: the current team data 
    - all_data_teams: DataFrame containing historical team data
    - teams_weights: dictionary containing weights for team parameters
    - teams_thresholds: dictionary containing thresholds for team parameters

    Returns:
    - teams_scores: a dictionary mapping game IDs to the calculated scores based on team comparison
    """
    teams_scores = {}

    for index, historical_teams in all_data_teams.iterrows():
        game_id = historical_teams['game_id']
        score = calculate_teams_score(current_teams, historical_teams, teams_weights, teams_thresholds)
        teams_scores[game_id] = score

    return teams_scores

def score_games_by_venues(current_venue, all_data_venues, venue_weights, venue_thresholds):
    """
    Calculate the score for each game in `all_data_venues` based on the similarity of the current venue with the historical venue.

    Parameters:
        current_venue (dict): A dictionary representing the current venue.
        all_data_venues (pandas.DataFrame): A DataFrame containing historical venue data.
        venue_weights (dict): A dictionary containing the weights for each venue parameter.
        venue_thresholds (dict): A dictionary containing the thresholds for each venue parameter.

    Returns:
        dict: A dictionary where the keys are the game IDs and the values are the scores.

    """
    venue_score = {}
    for index, historical_venue in all_data_venues.iterrows():
        game_id = historical_venue['game_id']
        score = calculate_venues_score(current_venue, historical_venue, venue_weights, venue_thresholds)
        venue_score[game_id] = score

    return venue_score

def combine_scores(game_id, score_dicts, category_weights):
    """
    Calculates the combined score for a specific game based on the provided score dictionaries and category weights.

    Parameters:
    - game_id: The ID of the game for which the score is being calculated
    - score_dicts: A dictionary containing scores for different categories
    - category_weights: A dictionary containing weights for different categories

    Returns:
    - combined_score: The total combined score for the specified game
    """
    combined_score = 0
    for category, scores in score_dicts.items():
        combined_score += scores.get(game_id, 0) * category_weights.get(category, 1)
    return combined_score

def get_combined_scores(score_dicts, category_weights):
    """
    Calculates the combined scores for different games based on the given score dictionaries and category weights.

    Parameters:
    - score_dicts: A dictionary containing scores for different categories
    - category_weights: A dictionary containing weights for different categories

    Returns:
    - combined_scores: A dictionary mapping game IDs to their calculated combined scores
    """
    combined_scores = {}
    all_game_ids = set()
    for scores in score_dicts.values():
        all_game_ids.update(scores.keys())
    
    for game_id in all_game_ids:
        combined_scores[game_id] = combine_scores(game_id, score_dicts, category_weights)
    
    return combined_scores

def assign_game_scores_to_plays(play_scores, combined_scores):
    """
    Assigns game scores to plays based on provided play_scores and combined_scores.

    Parameters:
    - play_scores: Dictionary containing play scores.
    - combined_scores: Dictionary containing combined scores for games.

    Returns:
    Updated play_scores dictionary after adding game scores to each play.
    """
    for index, (game_id, play_score) in play_scores.items():
        game_score = combined_scores.get(game_id, 0)
        play_scores[index] = (game_id, play_score + game_score)  # Add game score to each play
    return play_scores

def get_top_plays(play_scores, num_plays):
    """
    Given a dictionary of play scores and a number of plays to return, this function returns the top `num_plays` plays
    from the dictionary. The plays are sorted in descending order of their score. The function filters out any plays
    that are not followed by two consecutive plays of the same game_id.

    Parameters:
        play_scores (dict): A dictionary where the keys are the indices of the plays and the values are tuples of
        (play_type, play_score).
        num_plays (int): The number of top plays to return.

    Returns:
        list: A list of tuples, where each tuple contains the index of a play and its corresponding (play_type, play_score)
        tuple. The list is sorted in descending order of play_score.

    """
    sorted_plays = sorted(play_scores.items(), key=lambda x: x[1][1], reverse=True)
    filtered_plays = []

    for play in sorted_plays:
        current_index = play[0]
        
        if current_index + 2 < len(play_scores):
            next_play_1 = play_scores[current_index + 1]
            next_play_2 = play_scores[current_index + 2]
            
            if play[1][0] == next_play_1[0] == next_play_2[0]:
                filtered_plays.append(play)
                
                if len(filtered_plays) == num_plays:
                    break

    return filtered_plays[:num_plays]

def get_current_plays(top_plays, all_data_play_by_plays):
    """
    Given a list of top plays and a DataFrame of all play-by-play data, this function retrieves the current plays by checking if the index of the play exists in the DataFrame index. It returns a list of the current plays.

    Parameters:
        top_plays (list): A list of top plays.
        all_data_play_by_plays (DataFrame): DataFrame containing all play-by-play data.

    Returns:
        list: A list of current plays retrieved from the DataFrame based on the indices provided.
    """
    current_plays = []
    for index, _ in top_plays:
        if index in all_data_play_by_plays.index:
            previous_play = all_data_play_by_plays.loc[index]
            current_plays.append(previous_play)
    return current_plays

def get_next_plays(top_plays, all_data_play_by_plays):
    """
    Given a list of top plays and a DataFrame of all play-by-play data, this function retrieves the next plays by checking if the index of the play plus one exists in the DataFrame index. It returns a list of tuples, where each tuple contains the next play and its corresponding score from the top plays. If the next play's game ID is different from the previous play's game ID, it prints a message and waits for user input.

    Parameters:
        top_plays (list): A list of tuples, where each tuple contains the index of a play and its corresponding score.
        all_data_play_by_plays (pandas.DataFrame): A DataFrame containing all play-by-play data.

    Returns:
        list: A list of tuples, where each tuple contains the next play and its corresponding score from the top plays.
    """
    next_plays = []
    for index, score in top_plays:
        if index + 1 in all_data_play_by_plays.index:
            next_play = all_data_play_by_plays.loc[index + 1]
            if all_data_play_by_plays.loc[index, 'game_id'] != next_play['game_id']:
                # this should never happen
                print("next game id")
                input("wait")
            next_plays.append((next_play, score[1]))
    return next_plays

def get_next_next_plays(top_plays, all_data_play_by_plays):
    """
    Given a list of top plays and a DataFrame of all play-by-play data, this function retrieves the next two plays by checking if the index of the play plus two exists in the DataFrame index. It returns a list of DataFrame rows, where each row contains the next two plays. If the next two plays' game IDs are different from the previous play's game ID, it prints a message and waits for user input.

    Parameters:
        top_plays (list): A list of tuples, where each tuple contains the index of a play and its corresponding score.
        all_data_play_by_plays (pandas.DataFrame): A DataFrame containing all play-by-play data.

    Returns:
        list: A list of DataFrame rows, where each row contains the next two plays.
    """
    next_plays = []
    for index, _ in top_plays:
        if index + 2 in all_data_play_by_plays.index:
            next_play = all_data_play_by_plays.loc[index + 2]
            if all_data_play_by_plays.loc[index, 'game_id'] != next_play['game_id']:
                # this should never happen
                print("next next game id")
                input("wait")
            next_plays.append(next_play)
    return next_plays

def get_most_common_play_type(next_plays):
    """
    Calculates the most common play type in a list of next plays.

    Parameters:
    - next_plays (list): A list of tuples containing play dictionaries and scores.

    Returns:
    - str: The most common play type.
    """
    play_type_weights = Counter()
    for play, score in next_plays:
        play_type_weights[play['type']] += score  # Use score as weight
    
    most_common_play_type = play_type_weights.most_common(1)[0][0]
    return most_common_play_type

def analyze_next_plays(next_plays, most_common_play_type):
    """
    Analyzes the next plays and calculates the average stat yardage, average scoring play, and whether the start team and end team are different.

    Parameters:
        next_plays (list of tuples): A list of tuples containing the play and score.
        most_common_play_type (str): The most common play type.

    Returns:
        tuple: A tuple containing the average stat yardage, average scoring play, and a boolean indicating whether the start team and end team are different.
    """
    total_score = 0
    stat_yardage_weighted_sum = 0
    scoring_play_weighted_sum = 0
    start_team_end_team_different_count_weighted_sum = 0

    for play, score in next_plays:
        if play['type'] == most_common_play_type:
            total_score += score

            stat_yardage = play['statYardage']
            stat_yardage_weighted_sum += stat_yardage * score

            scoring_play = play['scoringPlay']
            scoring_play_weighted_sum += scoring_play * score

            start_team = play['start_team']
            end_team = play['end_team']
            if start_team != end_team:
                start_team_end_team_different_count_weighted_sum += score

    if total_score == 0:
        return 0, 0, False  # Handle case where there are no next plays

    average_stat_yardage = stat_yardage_weighted_sum / total_score
    average_scoring_play = scoring_play_weighted_sum / total_score
    start_team_end_team_different_higher_than_80_percent = start_team_end_team_different_count_weighted_sum / total_score > 0.8

    return average_stat_yardage, average_scoring_play, start_team_end_team_different_higher_than_80_percent

def analyze_clock(next_plays, next_next_plays, most_common_play_type):
    """
    Analyzes the clock time lost between plays, weighted by the play score.

    Parameters:
        next_plays (list of tuples): A list of tuples containing the play and score.
        next_next_plays (list of tuples): A list of tuples containing the next play and score.
        most_common_play_type (str): The most common play type.

    Returns:
        float: The average time lost between plays weighted by the play score.
    """
    total_time_lost_weighted_sum = 0
    total_score = 0

    for (next_play, score), next_next_play in zip(next_plays, next_next_plays):
        if next_play['type'] == most_common_play_type:
            time_next = next_play['clock']
            time_next_next = next_next_play['clock']

            period_next = next_play['period']
            period_next_next = next_next_play['period']
            if period_next == period_next_next:
                time_lost = time_next_next - time_next
            else:
                time_lost = ((900-time_next_next) + time_next) * -1

            play_score = score  # Use score as weight

            total_time_lost_weighted_sum += time_lost * play_score
            total_score += play_score

    if total_score == 0:
        return 0  # Handle case where there are no plays to compare

    average_time_lost_weighted = total_time_lost_weighted_sum / total_score
    return average_time_lost_weighted

def analyze_score(current_plays, next_plays, most_common_play_type):
    """
    Analyzes the score difference between current and next plays, weighted by the play score.

    Parameters:
        current_plays (list): A list of current plays.
        next_plays (list of tuples): A list of tuples containing the next play and score.
        most_common_play_type (str): The most common play type.

    Returns:
        float: The average score difference weighted by the play score. Returns 0 if there are no plays to compare.
    """
    total_score_difference_weighted_sum = 0
    total_score = 0

    for current_play, (next_play, score) in zip(current_plays, next_plays):
        if next_play['type'] == most_common_play_type:
            current_score = current_play['awayScore'] + current_play['homeScore']
            next_score = next_play['awayScore'] + next_play['homeScore']
            play_score = score  # Use score as weight

            score_difference = next_score - current_score
            total_score_difference_weighted_sum += score_difference * play_score
            total_score += play_score

    if total_score == 0:
        return 0  # Handle case where there are no plays to compare

    average_score_difference_weighted = total_score_difference_weighted_sum / total_score 
    return average_score_difference_weighted

def get_position_combinations(next_plays, all_data_players, most_common_play_type):
    """
    Calculates the position combinations based on the most common play type.

    Parameters:
    - next_plays (list of tuples): A list of tuples containing play dictionaries and scores.
    - all_data_players (DataFrame): A DataFrame containing player data.
    - most_common_play_type (str): The most common play type.

    Returns:
    - list: A list of tuples containing position combinations and scores.
    """
    position_combinations = []
    for play, score in next_plays:
        if play['type'] == most_common_play_type:
            player_ids = play['text'].split()
            positions = []
            for player_id in player_ids:
                position = all_data_players.loc[all_data_players['player_id'] == int(player_id), 'position_id'].iloc[0]
                positions.append(position)
            positions.sort()  # Sort positions to treat [QB, WR] the same as [WR, QB]
            position_combinations.append((tuple(positions), score))
    return position_combinations

def get_most_common_position_combination(position_combinations):
    """
    Calculates the most common position combination in a list of positions and scores.

    Parameters:
    - position_combinations (list): A list of tuples containing position combinations and scores.

    Returns:
    - tuple: The most common position combination.
    """
    position_counter = Counter()
    for positions, score in position_combinations:
        position_counter[positions] += score
    return position_counter.most_common(1)[0][0]  # Get the most common position combination

def calculate_average_player_attributes(players_data, game_dates):
    """
    Calculates the average weight, height, and age of players based on their data and game dates.

    Parameters:
    - players_data (DataFrame): A DataFrame containing player data including weight, height, age, score, college_id, etc.
    - game_dates (DataFrame): A DataFrame containing game dates associated with each player's game.

    Returns:
    - Tuple: A tuple containing the average weight, average height, average age, and a dictionary of college scores.
    """
    # Merge game_dates to get the game date for each player
    players_data = players_data.merge(game_dates, on='game_id', how='left')
    
    # Calculate age in days for each player at the time of the game
    players_data['age'] = players_data['days_since_epoch'] - players_data['date_of_birth']

    # Initialize variables to accumulate weighted sums
    total_weighted_weight = 0
    total_weighted_height = 0
    total_weighted_age = 0
    college_counter = defaultdict(int)

    total_score = 0
    
    for index, player_data in players_data.iterrows():
        score = player_data['score']
        total_weighted_weight += player_data['weight'] * score
        total_weighted_height += player_data['height'] * score
        total_weighted_age += player_data['age'] * score
        college_counter[player_data['college_id']] += score
        total_score += score
    
    # Calculate weighted averages
    if total_score > 0:
        average_weight = total_weighted_weight / total_score
        average_height = total_weighted_height / total_score
        average_age = total_weighted_age / total_score
        college_score = {college_id: count / total_score for college_id, count in college_counter.items()}
    else:
        average_weight = 0
        average_height = 0
        average_age = 0
        college_score = {}

    return average_weight, average_height, average_age, college_score

def get_player_id_occurrences(next_plays, most_common_positions, all_data_players, all_data_dates, most_common_play_type):
    """
    Calculates the occurrences of player IDs and their positions in the next plays, and returns the average player attributes for each position.

    Parameters:
    - next_plays (list of tuples): A list of tuples containing the next plays and their corresponding scores.
    - most_common_positions (list): A list of the most common positions in the next plays.
    - all_data_players (pandas.DataFrame): A DataFrame containing player data.
    - all_data_dates (pandas.DataFrame): A DataFrame containing game dates.
    - most_common_play_type (str): The most common play type in the next plays.

    Returns:
    - position_averages (dict): A dictionary containing the average player attributes for each position.
    - player_counter (defaultdict): A defaultdict containing the count of player IDs and their corresponding scores.

    The function iterates over the next plays and checks if the play type matches the most common play type. If it does, it splits the play text into player IDs and retrieves their positions from the all_data_players DataFrame. If the positions match the most common positions, it increments the player counter and appends the player ID and score to the position_player_data dictionary.

    After processing all the next plays, the function calculates the average player attributes for each position by creating a DataFrame of player data and calling the calculate_average_player_attributes function. The average player attributes are stored in the position_averages dictionary.

    Finally, the function returns the position_averages dictionary and the player_counter defaultdict.
    """
    player_counter = defaultdict(int)
    position_player_data = defaultdict(list)
    
    for play, score in next_plays:
        if play['type'] == most_common_play_type:
            player_ids = play['text'].split()
            positions = []
            for player_id in player_ids:
                position = all_data_players.loc[all_data_players['player_id'] == int(player_id), 'position_id'].iloc[0]
                positions.append(position)
            positions.sort()
            if positions == list(most_common_positions):
                for player_id in player_ids:
                    player_counter[player_id] += score
                    position = all_data_players.loc[all_data_players['player_id'] == int(player_id), 'position_id'].iloc[0]
                    position_player_data[position].append((player_id, score))
    
    game_dates = all_data_dates.set_index('game_id')['days_since_epoch']
    position_averages = {}
    for position, player_idsANDscores in position_player_data.items():
        players_data = []
        for player_id, score in player_idsANDscores:
            player_data = all_data_players.loc[all_data_players['player_id'] == int(player_id)].iloc[0]
            player_data['score'] = score
            players_data.append(player_data)

        players_data_df = pd.DataFrame(players_data)
        avg_weight, avg_height, avg_age, college_score = calculate_average_player_attributes(players_data_df, game_dates)
        position_averages[position] = {
            'average_weight': avg_weight,
            'average_height': avg_height,
            'average_age': avg_age,
            'college_score': college_score
        }

    
    return position_averages, player_counter

def select_best_match_players(most_common_positions, game_prediction_players, game_prediction_teams, team, player_counter, position_averages, all_data_dates):
    """
    Selects the best match players based on the most common positions, game prediction players, game prediction teams, team, player counter, position averages, and all data dates.

    Parameters:
    - most_common_positions (list): A list of the most common positions.
    - game_prediction_players (DataFrame): A DataFrame containing game prediction players.
    - game_prediction_teams (DataFrame): A DataFrame containing game prediction teams.
    - team (int): The team ID.
    - player_counter (dict): A dictionary containing player counters.
    - position_averages (dict): A dictionary containing position averages.
    - all_data_dates (DataFrame): A DataFrame containing all data dates.

    Returns:
    - list: A list of the best player IDs.
    """
    home_id = game_prediction_teams['home_id'].iloc[0]
    away_id = game_prediction_teams['away_id'].iloc[0]
    
    def calculate_score(player, position_averages, player_weights, player_thresholds, all_data_dates):
        position_avg = position_averages[player['position_id']]
        score = 0
        game_id = player['game_id']
        game_dates = all_data_dates.set_index('game_id')['days_since_epoch']
        days_since_epoch_of_game = game_dates.loc[game_id]

        for key in player_weights:
            if key == 'college_id':
                college_id = player['college_id']
                if college_id in position_avg['college_score']:
                    frequency = position_avg['college_score'][college_id]
                    score += player_weights['college_id'] * frequency
            if key == 'height':
                if (abs(player['height'] - position_avg['average_height'])) <= player_thresholds['height']:
                    score += player_weights['height']
            if key == 'weight':
                if (abs(player['weight'] - position_avg['average_weight'])) <= player_thresholds['weight']:
                    score += player_weights['weight']
            if key == 'age':
                player_age = days_since_epoch_of_game - player['date_of_birth']
                if (abs(player_age - position_avg['average_age'])) <= player_thresholds['age']:
                    score += player_weights['age']
        return score
    
    best_players = []

    for pos_id in most_common_positions:
        if pos_id in [1, 7, 8, 9, 10, 22]:
            team_id = team
        else:
            if team == home_id:
                team_id = away_id
            else:
                team_id = home_id
        
        candidates = game_prediction_players[(game_prediction_players['team_id'] == team_id) & (game_prediction_players['position_id'] == pos_id)].copy()
        if candidates.empty:
            continue
        
        if len(candidates) > 1:
            candidate_ids = candidates['player_id'].tolist()
            player_counter_keys = list(map(str, player_counter.keys()))  # Convert player_counter keys to strings
            common_players = [player_id for player_id in candidate_ids if str(player_id) in player_counter_keys]
            if common_players:
                best_player_id = max(common_players, key=lambda player_id: player_counter.get(str(player_id), 0))
            else:
                candidates.loc[:, 'score'] = candidates.apply(lambda player: calculate_score(player, position_averages, player_weights, player_thresholds, all_data_dates), axis=1)
                best_player_id = candidates.loc[candidates['score'].idxmax(), 'player_id']
        else:
            best_player_id = candidates.iloc[0]['player_id']
        
        best_players.append(best_player_id)
    
    return best_players

def new_previous_play(previousPlay, most_common_play_type, best_players, start_team_end_team_different_higher_than_50_percent, game_prediction_teams, avg_score_diff, avg_clock_diff, average_scoring_play, average_stat_yardage, kickoff): 
        """
        Updates the previous play based on various game conditions and predictions.

        Parameters:
        - previousPlay (dict): The dictionary containing information about the previous play.
        - most_common_play_type (str): The most common play type.
        - best_players (str): The best players for the play.
        - start_team_end_team_different_higher_than_50_percent (bool): Indicates if the start and end teams are different by more than 50%.
        - game_prediction_teams (DataFrame): A DataFrame containing team prediction data.
        - avg_score_diff (float): The average score difference.
        - avg_clock_diff (int): The average clock difference.
        - average_scoring_play (int): The average scoring play.
        - average_stat_yardage (int): The average stat yardage.
        - kickoff (dict): Information about the kickoff.

        Returns:
        - None
        """
        lastPlayScoring = previousPlay['scoringPlay']
        addedScore = False

        if previousPlay['period'] == 2 and previousPlay['clock'] == 0 and previousPlay['type'] != 'Kickoff':
            # 2nd half kickoff
            if kickoff['start_team'] == game_prediction_teams['home_id'].iloc[0]:
                start_team = game_prediction_teams['away_id'].iloc[0]
                end_team = game_prediction_teams['home_id'].iloc[0]
            else:
                start_team = game_prediction_teams['home_id'].iloc[0]
                end_team = game_prediction_teams['away_id'].iloc[0]

                    
            kicker = random.choice(game_prediction_players[(game_prediction_players['position_id'] == 22) & (game_prediction_players['team_id'] == start_team)]['player_id'].tolist())

            previousPlay['type'] = 'Kickoff'
            previousPlay['text'] = kicker
            previousPlay['scoringPlay'] = 0
            previousPlay['start_down'] = -1
            previousPlay['start_distance'] = -1
            previousPlay['start_yardLine'] = 0
            previousPlay['start_yardsToEndzone'] = 65
            previousPlay['start_team'] = start_team
            previousPlay['end_team'] = end_team
            previousPlay['end_down'] = 1
            previousPlay['end_distance'] = 10
            previousPlay['end_yardLine'] = 0
            previousPlay['end_yardsToEndzone'] = 75
            previousPlay['statYardage'] = 0
            previousPlay['period'] = 3
            previousPlay['clock'] = 890

        else:
        
            start_team = previousPlay['end_team']
            end_team = start_team

            previousPlay['type'] = most_common_play_type
            previousPlay['text'] = best_players
            if previousPlay['type'] == 'Pass Incompletion':
                # this is needed because espns play by play gives stat yardage for incompletions
                # (on accident???) sometimes
                average_stat_yardage = 0
            
            previousPlay['statYardage'] = average_stat_yardage

            if previousPlay['type'] == 'Field Goal Good':
                average_scoring_play = 1
                avg_score_diff = 3

            if previousPlay['type'] == 'Field Goal Missed':
                average_scoring_play = 0
                avg_score_diff = 0

            if end_team == game_prediction_teams['away_id'].iloc[0]:
                if avg_score_diff > 0 and avg_score_diff <= 7.1 and average_scoring_play > 0.0001:
                    previousPlay['awayScore'] = previousPlay['awayScore'] + avg_score_diff
                    previousPlay['scoringPlay'] = average_scoring_play
                    addedScore = True

            if end_team == game_prediction_teams['home_id'].iloc[0]:
                if avg_score_diff > 0 and avg_score_diff <= 7.1 and average_scoring_play > 0.0001:
                    previousPlay['homeScore'] = previousPlay['homeScore'] + avg_score_diff
                    previousPlay['scoringPlay'] = average_scoring_play
                    addedScore = True

            if previousPlay['clock'] == 0:
                previousPlay['period'] = previousPlay['period'] + 1
                previousPlay['clock'] = 900
            else:
                if (previousPlay['clock'] + avg_clock_diff) < 0:
                    previousPlay['clock'] = 0
                else:
                    previousPlay['clock'] = previousPlay['clock'] + avg_clock_diff

            if addedScore:
                previousPlay['scoringPlay'] = average_scoring_play
            else:
                previousPlay['scoringPlay'] = 0

            previousPlay['start_down'] = previousPlay['end_down']
            previousPlay['start_distance'] = previousPlay['end_distance']
            previousPlay['start_yardsToEndzone'] = previousPlay['end_yardsToEndzone']

            if start_team_end_team_different_higher_than_50_percent:
                if start_team == game_prediction_teams['home_id'].iloc[0]:
                    end_team = game_prediction_teams['away_id'].iloc[0]
                else:
                    end_team = game_prediction_teams['home_id'].iloc[0]

                #assuming that change of possesion on average results in this
                previousPlay['end_down'] = 1
                previousPlay['end_distance'] = 10
                if previousPlay['type'] == 'Field Goal Missed':
                    previousPlay['end_yardsToEndzone'] = 100 - (previousPlay['end_yardsToEndzone'])
                else:
                    previousPlay['end_yardsToEndzone'] = 70

            else:

                if (previousPlay['end_distance'] - average_stat_yardage) < 0:
                    previousPlay['end_down'] = 1
                    previousPlay['end_yardsToEndzone'] = previousPlay['end_yardsToEndzone'] - average_stat_yardage
                    if previousPlay['end_yardsToEndzone'] < 10:
                        previousPlay['end_distance'] = previousPlay['end_yardsToEndzone']
                    else:
                        previousPlay['end_distance'] = 10
                else:
                    if (previousPlay['end_down'] == 4 and addedScore == False):
                        # turn over on downs
                        previousPlay['end_down'] = 1
                        previousPlay['end_distance'] = 10
                        previousPlay['end_yardsToEndzone'] = 100 - (previousPlay['end_yardsToEndzone'] - average_stat_yardage)
                        if start_team == game_prediction_teams['home_id'].iloc[0]:
                            end_team = game_prediction_teams['away_id'].iloc[0]
                        else:
                            end_team = game_prediction_teams['home_id'].iloc[0]
                    else:
                            previousPlay['end_down'] = previousPlay['end_down'] + 1
                            previousPlay['end_distance'] = previousPlay['end_distance'] - average_stat_yardage
                            previousPlay['end_yardsToEndzone'] = previousPlay['end_yardsToEndzone'] - average_stat_yardage

            if (previousPlay['end_yardsToEndzone'] <= 0 or previousPlay['scoringPlay'] > 0.75):
                #touchdown
                previousPlay['scoringPlay'] = 1
                previousPlay['statYardage'] = previousPlay['start_yardsToEndzone']
                previousPlay['end_yardsToEndzone'] = 0
                previousPlay['end_down'] = -1
                previousPlay['end_distance'] = -1
                if end_team == game_prediction_teams['home_id'].iloc[0]:
                    if (addedScore == False) and previousPlay['start_down'] != -1 and (previousPlay['type'] != 'Field Goal Good' and previousPlay['type'] != 'Field Goal Missed'):
                        previousPlay['homeScore'] = previousPlay['homeScore'] + 6
                else:
                    if (addedScore == False) and previousPlay['start_down'] != -1 and (previousPlay['type'] != 'Field Goal Good' and previousPlay['type'] != 'Field Goal Missed'):
                        previousPlay['awayScore'] = previousPlay['awayScore'] + 6

            if previousPlay['scoringPlay'] > 0.5:
                previousPlay['end_yardsToEndzone'] = 0
            
            previousPlay['start_team'] = start_team
            previousPlay['end_team'] =  end_team

            if previousPlay['type'] == 'Penalty':
                previousPlay['end_down'] = previousPlay['start_down']
                previousPlay['end_distance'] = previousPlay['start_distance'] - average_stat_yardage
                previousPlay['scoringPlay'] = lastPlayScoring
                previousPlay['end_team'] = previousPlay['start_team']
                previousPlay['end_yardsToEndzone'] = previousPlay['start_yardsToEndzone']

weather_thresholds = {
    'temp_max': 5, 'temp_min': 5, 'temp': 5, 'feels_like_max': 5, 'feels_like_min': 5, 'feels_like': 5,
    'dew': 5, 'humidity': 10, 'precipitation': 1, 'precipitation_probability': 10, 'precipitation_coverage': 10,
    'snow': 1, 'snow_depth': 1, 'wind_gust': 5, 'wind_speed': 5, 'wind_direction': 10, 'pressure': 10,
    'cloud_cover': 10, 'visibility': 1, 'solar_radiation': 10, 'solar_energy': 10, 'uv_index': 1,
    'severe_risk': 1, 'sunrise_epoch': 600, 'sunset_epoch': 600, 'moon_phase': 0
}

weather_weights = {
    'temp_max': 1.5, 'temp_min': 1.5, 'temp': 1.5, 'feels_like_max': 1.5, 'feels_like_min': 1.5, 'feels_like': 1.5,
    'dew': 0.25, 'humidity': 0.75, 'precipitation': 2, 'precipitation_probability': 2, 'precipitation_coverage': 1,
    'snow': 5, 'snow_depth': 5, 'wind_gust': 4, 'wind_speed': 4, 'wind_direction': 1.5, 'pressure': 1,
    'cloud_cover': 0.5, 'visibility': 0.5, 'solar_radiation': 0.25, 'solar_energy': 0.25, 'uv_index': 0.5,
    'severe_risk': 1, 'sunrise_epoch': 0.25, 'sunset_epoch': 0.25, 'moon_phase': 1
}

play_thresholds = {
    'type': 0, 'awayScore': 3, 'homeScore': 3, 'period': 0, 'clock': 120, 'scoringPlay': 0.25,
    'start_down': 0, 'start_distance': 3, 'start_yardLine': 0, 'start_yardsToEndzone': 10,
    'end_down': 0, 'end_distance': 2.5, 'end_yardLine': 0, 'end_yardsToEndzone': 10.5, 'statYardage': 25,
    'start_team': 0, 'end_team': 0
}

play_weights = {
    'type': 57.5, 'awayScore': 5, 'homeScore': 5, 'period': 12.5, 'clock': 12.5, 'scoringPlay': 49.5,
    'start_down': 12, 'start_distance': 12, 'start_yardLine': 0, 'start_yardsToEndzone': 14.5,
    'end_down': 50.5, 'end_distance': 45.5, 'end_yardLine': 0, 'end_yardsToEndzone': 59.25, 'statYardage': 3,
    'start_team': 13, 'end_team': 13
}

date_thresholds = {
    'one_hour': 3600,  # one hour in seconds
    'eight_months': 240,  # eight months in days
    'same_weekday': 0  # placeholder, no threshold needed for this criterion
}

date_weights = {
    'one_hour': 2,
    'eight_months': 38.5,
    'same_weekday': 2
}

game_thresholds = {
    'season_type': 0,
    'week_num': 1
}

game_weights = {
    'season_type': 20,
    'week_num': 5
}

team_thresholds = {
    'home_id': 0,
    'home_id_is_away': 0,
    'away_id': 0,
    'away_id_is_home': 0,
}

team_weights = {
    'home_id': 15,
    'home_id_is_away': 5,
    'away_id': 15,
    'away_id_is_home': 5,
}

venue_thresholds = {
    'venue_id': 0,
    'grass': 0,
    'indoor': 0
}

venue_weights = {
    'venue_id': 25,
    'grass': 7.5,
    'indoor': 10
}

player_thresholds = {
    'weight': 10,
    'height': 1.5,
    'age': 730,
    'college_id': 0
}

player_weights = {
    'weight': 10,
    'height': 10,
    'age': 10,
    'college_id': 10
}

category_weights = {
    'weather': 0.5,
    'dates': 1,
    'games': 0.2,
    'teams': 0.65,
    'venues': 0.5
}

# Load all historical data
all_data_games = get_all_data('Games')
all_data_dates = get_all_data('Dates')
all_data_teams = get_all_data('Teams')
all_data_venue = get_all_data('Venues')
all_data_weather = get_all_data('Weather')
all_data_players = get_all_data('Players')
all_data_play_by_plays = get_all_data('PlayByPlays').sort_values(by=['game_id', 'period', 'clock'], ascending=[True, True, False])

# query not needed to get games that don't have a prediction because we already loaded all the date from the 'Games' table
games_with_false_pred = all_data_games[all_data_games['has_pred'] == False]

# loop through every game that doesn't have a prediction, and get a prediction
for game_id in games_with_false_pred['game_id']:

    # Load the data for the game we are trying to predict
    game_prediction_games = get_data_for_game_id(game_id, 'Games')
    game_prediction_dates = get_data_for_game_id(game_id, 'Dates')
    game_prediction_teams = get_data_for_game_id(game_id, 'Teams')
    game_prediction_venue = get_data_for_game_id(game_id, 'Venues')
    game_prediction_weather = get_data_for_game_id(game_id, 'Weather')
    game_prediction_players = get_data_for_game_id(game_id, 'Players')

    # Remove any plays that have come after the game we are trying to predict, and also remove unnecessary plays like timeouts, end of quarter, etc...
    game_prediction_days_since_epoch = game_prediction_dates['days_since_epoch'].values[0]
    filtered_game_ids = all_data_dates[all_data_dates['days_since_epoch'] >= game_prediction_days_since_epoch]['game_id'].unique()
    all_data_play_by_plays = all_data_play_by_plays[~all_data_play_by_plays['game_id'].isin(filtered_game_ids)]
    conditions = (
        ~all_data_play_by_plays['type'].str.contains('timeout', case=False) & 
        ~all_data_play_by_plays['type'].str.contains('End Period', case=False) & 
        ~all_data_play_by_plays['type'].str.contains('End of Half', case=False) & 
        ~all_data_play_by_plays['type'].str.contains('Coin Toss', case=False) & 
        ~all_data_play_by_plays['type'].str.contains('Two-minute warning', case=False)
    )
    all_data_play_by_plays = all_data_play_by_plays[conditions]

    # MUST RESET INDEX OF PLAYS SINCE SOME HAVE BEEN REMOVED. VERY VERY IMPORTANT!!!!
    all_data_play_by_plays.reset_index(drop=True, inplace=True)

    # Pick a random team and a random kicker from that team to kick the ball off in our game
    start_team, end_team = random.sample([game_prediction_teams['home_id'].iloc[0], game_prediction_teams['away_id'].iloc[0]], 2)
    kicker = random.choice(game_prediction_players[(game_prediction_players['position_id'] == 22) & (game_prediction_players['team_id'] == start_team)]['player_id'].tolist())

    # Initialize the first play of each game which will always be a kickoff.
    previousPlayType = 'Kickoff'
    previousPlayText = kicker
    previousPlayAwayScore = 0
    previousPlayHomeScore = 0
    previousPlayPeriod = 1
    previousPlayClock = 900
    previousPlayScoringPlay = False
    previousPlayStartDown = -1
    previousPlayStartDistance = -1
    previousPlayStartYardLine = 0 #yardline doesnt matter
    previousPlayStartYardsToEndzone = 65
    previousPlayStartTeam = start_team
    previousPlayEndTeam = end_team
    previousPlayEndDown = 1
    previousPlayEndDistance = 10
    previousPlayEndYardLine = 0 #yardline doesnt matter
    previousPlayEndYardsToEndzone = 75 # assuming most kickoffs will be a touchback or returned to around the 25 yardline
    previousPlayStatYardage = 0

    previousPlay = {
        'type': previousPlayType,
        'text': previousPlayText,
        'awayScore': previousPlayAwayScore,
        'homeScore': previousPlayHomeScore,
        'period': previousPlayPeriod,
        'clock': previousPlayClock,
        'scoringPlay': previousPlayScoringPlay,
        'start_down': previousPlayStartDown,
        'start_distance': previousPlayStartDistance,
        'start_yardLine': previousPlayStartYardLine,
        'start_yardsToEndzone': previousPlayStartYardsToEndzone,
        'start_team': previousPlayStartTeam,
        'end_team': previousPlayEndTeam,
        'end_down': previousPlayEndDown,
        'end_distance': previousPlayEndDistance,
        'end_yardLine': previousPlayEndYardLine,
        'end_yardsToEndzone': previousPlayEndYardsToEndzone,
        'statYardage': previousPlayStatYardage
    }

    # store the kickoff information in a seperate variable since previousPlay is going to change.
    # 1st half kickoff information will be used to determine who kicks off in the second half
    kickoff = {
        'type': previousPlayType,
        'text': previousPlayText,
        'awayScore': previousPlayAwayScore,
        'homeScore': previousPlayHomeScore,
        'period': previousPlayPeriod,
        'clock': previousPlayClock,
        'scoringPlay': previousPlayScoringPlay,
        'start_down': previousPlayStartDown,
        'start_distance': previousPlayStartDistance,
        'start_yardLine': previousPlayStartYardLine,
        'start_yardsToEndzone': previousPlayStartYardsToEndzone,
        'start_team': previousPlayStartTeam,
        'end_team': previousPlayEndTeam,
        'end_down': previousPlayEndDown,
        'end_distance': previousPlayEndDistance,
        'end_yardLine': previousPlayEndYardLine,
        'end_yardsToEndzone': previousPlayEndYardsToEndzone,
        'statYardage': previousPlayStatYardage
    }

    # score the all the historical games based on how similar the following are to the current game we are trying to predict
    weather_scores = score_games_by_weather(game_prediction_weather.iloc[0], all_data_weather, weather_weights, weather_thresholds)

    date_scores = score_games_by_date(game_prediction_dates.iloc[0], all_data_dates, date_weights, date_thresholds)

    game_scores = score_games_by_games(game_prediction_games.iloc[0], all_data_games, game_weights, game_thresholds)

    team_scores = score_games_by_teams(game_prediction_teams.iloc[0], all_data_teams, team_weights, team_thresholds)

    venue_scores = score_games_by_venues(game_prediction_venue.iloc[0], all_data_venue, venue_weights, venue_thresholds)

    score_dicts = {
        'weather': weather_scores,
        'dates': date_scores,
        'games': game_scores,
        'teams': team_scores,
        'venues': venue_scores
    }

    all_plays = []

    # keep looping and predicting the next play until the game is over
    while not (previousPlay['period'] == 4 and previousPlay['clock'] == 0):
        print(game_id)
        print(previousPlay['period'])
        print(previousPlay['clock'])
        
        all_plays.append(previousPlay.copy())

        # score all the historical plays based on how similar it was to the previous play
        play_scores = score_plays(previousPlay, all_data_play_by_plays, play_weights, play_thresholds)

        # combine the weather, date, game, team, and venue score, and assign each game_id their score
        combined_scores = get_combined_scores(score_dicts, category_weights)

        # combine each play score with the score of its respective game_id to get a final score for each play
        play_scores_with_game_scores = assign_game_scores_to_plays(play_scores, combined_scores)

        # get the index top n most similar plays
        n = 5
        top_plays = get_top_plays(play_scores_with_game_scores, n)

        # get the acutal top n most similar plays
        current_plays = get_current_plays(top_plays, all_data_play_by_plays)

        # get the play that came directly after each of the top/current plays
        next_plays = get_next_plays(top_plays, all_data_play_by_plays)

        # get the 2nd play that came directly after each of the top/current plays (this is needed to calculate how long each play took)
        next_next_plays = get_next_next_plays(top_plays, all_data_play_by_plays)

        # from the next play, get the most common play type (run, pass, punt, etc...)
        most_common_play_type = get_most_common_play_type(next_plays)

        # for all the next plays that are of the most common play type, calculate the average yards gained/lost, if its a scoring play, and if possesion changed
        average_stat_yardage, average_scoring_play, start_team_end_team_different_higher_than_50_percent = analyze_next_plays(next_plays, most_common_play_type)

        # for all the next plays that are of the most common play type, calculate how long they took
        avg_clock_diff = analyze_clock(next_plays, next_next_plays, most_common_play_type)

        # for all the next plays that are of the most common play type, calculate how many points are typically scored
        avg_score_diff = analyze_score(current_plays, next_plays, most_common_play_type)

        # for all the next plays that are of the most common play type, get the different kinds of positions involved in the play 
        position_combinations = get_position_combinations(next_plays, all_data_players, most_common_play_type)

        # get the most common combination of positions
        most_common_positions = get_most_common_position_combination(position_combinations)
        
        # get the average height, weight, age, and college id for each position, and also a count of what player id's are in those plays
        position_averages, player_counter = get_player_id_occurrences(next_plays, most_common_positions, all_data_players, all_data_dates, most_common_play_type)
        
        # determine the most likely players to be involved in the next play. Each position is assigned a player, based on player_counter and position_averages
        best_players = select_best_match_players(most_common_positions, game_prediction_players, game_prediction_teams, previousPlay['end_team'], player_counter, position_averages, all_data_dates)

        # using all the information from above, create the next play
        newPlay = new_previous_play(previousPlay, most_common_play_type, best_players, start_team_end_team_different_higher_than_50_percent, game_prediction_teams, avg_score_diff, avg_clock_diff, average_scoring_play, average_stat_yardage, kickoff)
        
        # save the newest play, and run the loop again, now with the new play being the old play
        df_all_plays = pd.DataFrame(all_plays)
        df_all_plays.to_csv('all_plays.csv', index=False)


    # append the last play, since the loop stopped iterating
    all_plays.append(previousPlay.copy())
    df_all_plays = pd.DataFrame(all_plays)
    df_all_plays.to_csv('all_plays.csv', index=False)
    input("wait")
    
    # update sql table
    with engine.connect() as connection:
        transaction = connection.begin()
        try:
            # Step 2: Update the Games table and delete from the Predictions table
            update_query = text("""
            UPDATE Games
            SET has_pred = TRUE
            WHERE game_id = :game_id
            """)
            delete_query = text("DELETE FROM Predictions WHERE game_id = :game_id")
            
            connection.execute(update_query, {'game_id': game_id})
            connection.execute(delete_query, {'game_id': game_id})
            
            transaction.commit()
            print(f"Updated Games and deleted Predictions for game_id: {game_id}")
        except Exception as e:
            transaction.rollback()
            print(f"Transaction failed: {e}")


    # clean up the df so it fits with the sql table, then send to sql
    df_all_plays['game_id'] = game_id
    df_all_plays['id'] = 0
    df_all_plays['team_id'] = 0
    df_all_plays['text'] = df_all_plays['text'].apply(lambda x: str(x) if isinstance(x, list) else x)
    df_all_plays.to_sql('Predictions', con=engine, if_exists='append', index=False)



    end_time = time.time()
    duration = end_time - start_time
    print(f"Time taken for the loop: {duration} seconds")


# 1 to 30 check pass incompletion 19 yard gain