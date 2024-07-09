-- Update neural_network_seen column in the Games table to TRUE
-- where game_id matches those in the Venues table with NULL zip_code

UPDATE NFL.Games
SET neural_network_seen = TRUE
WHERE game_id IN (
    SELECT game_id
    FROM NFL.Venues
    WHERE zip_code IS NULL
);
