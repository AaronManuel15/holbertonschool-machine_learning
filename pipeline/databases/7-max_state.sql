-- computes the max temp for all states
SELECT state, MAX(value) AS max_temp FROM temperatures
GROUP BY state
ORDER BY STATE ASC;
