-- lists all shows from the database with their rating sum
-- SELECT * FROM tv_shows;
-- SELECT * FROM tv_genres;
-- SELECT * FROM tv_show_genres;
-- SELECT * FROM tv_show_ratings;

SELECT tv_shows.title AS title, SUM(tv_show_ratings.rate) AS rating 
FROM tv_shows
LEFT JOIN tv_show_ratings ON tv_shows.id=tv_show_ratings.show_id
GROUP BY tv_shows.title
ORDER BY rating DESC
