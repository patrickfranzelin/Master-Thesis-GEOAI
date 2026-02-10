-- =========================================
-- OpenBuildings incremental import
-- CSV must be at: C:/temp/open_buildings.csv
--& "C:\Program Files\PostgreSQL\18\bin\psql.exe" `
--  -U postgres `
--  -d geoai `
--  -f "C:\git\Master-Thesis-GEOAI\scripts\sql_script\import_open_buildings.sql"

-- =========================================

BEGIN;

CREATE EXTENSION IF NOT EXISTS postgis;
CREATE SCHEMA IF NOT EXISTS src;

-- -----------------------------------------
-- 1. Staging table (always recreated)
-- -----------------------------------------

DROP TABLE IF EXISTS src._staging_open_buildings;

CREATE TABLE src._staging_open_buildings (
    latitude double precision,
    longitude double precision,
    area_m2 double precision,
    confidence real,
    geom text,
    plus_code text
);

-- -----------------------------------------
-- 2. Load CSV (CLIENT SIDE!)
-- -----------------------------------------
-- IMPORTANT: this is a psql command, not pure SQL

\copy src._staging_open_buildings FROM 'C:\temp\11d_buildings.csv' CSV HEADER;

-- -----------------------------------------
-- 3. Insert into main buildings table
-- geom is MultiPolygon
-- tiff_path stays NULL
-- -----------------------------------------

-- -----------------------------------------
-- 3. Insert ONLY buildings intersecting a TIFF
-- -----------------------------------------

INSERT INTO src.buildings (geom, area_m2, confidence, plus_code, tiff_path)
SELECT
    ST_GeomFromText(s.geom, 4326) AS geom,
    s.area_m2,
    s.confidence,
    s.plus_code,
    t.path AS tiff_path
FROM src._staging_open_buildings s
JOIN src.tiffs t
  ON ST_Intersects(
        ST_GeomFromText(s.geom, 4326),
        t.geometry
     )
WHERE s.geom IS NOT NULL;


-- -----------------------------------------
-- 4. Cleanup
-- -----------------------------------------

DROP TABLE src._staging_open_buildings;

COMMIT;

-- -----------------------------------------
-- 5. Report
-- -----------------------------------------

SELECT COUNT(*) AS total_buildings FROM src.buildings;
