-- Task 16: Old school band
SELECT band_name, IFNULL((split-formed), (2020 - formed)) AS lifespan
FROM metal_bands
WHERE style LIKE '%Glam rock%'
ORDER BY lifespan DESC;
