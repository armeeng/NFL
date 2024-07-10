-- Delete plays from PlayByPlays where game_id matches those in the Venues table with NULL zip_code

DELETE FROM NFL.PlayByPlays
WHERE game_id IN (
    SELECT game_id
    FROM NFL.Venues
    WHERE zip_code IS NULL
);

