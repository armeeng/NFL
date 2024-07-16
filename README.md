Introduction
This project leverages historical NFL game data to predict future game plays. It interacts with ESPN's API and a MySQL database to fetch, process, and store game data, which is then used to make predictions.

Scripts Overview
NFL_GET_GAMEID: Collecting Game Metadata
Purpose: Collects game metadata from ESPN's API and stores it in the database.
NFL_GET_FINISHED: Checking Game Status
Purpose: Checks if the games have finished and updates the game_finished status in the database.
NFL_GET_INPUTS: Collecting Detailed Game Inputs
Purpose: Collects detailed game inputs such as team rosters, game dates, and weather information, and stores them in the database.
NFL_GET_PBP: Collecting Play-by-Play Data
Purpose: Collects play-by-play data for finished games and stores it in the database.
NFL_GET_PRED: Predicting Play-by-Play Outcomes
Purpose: Predicts the play-by-play outcomes for future games based on historical data and stores it in the database.
