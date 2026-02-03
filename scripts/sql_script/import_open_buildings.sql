-- =========================================
-- OpenBuildings incremental import
-- CSV must be at: C:/temp/open_buildings.csv
-- "C:\Program Files\PostgreSQL\18\bin\psql.exe" -U postgres -d geoai -f C:\path\to\import_open_buildings.sql

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

\copy src._staging_open_buildings FROM 'C:/temp/open_buildings.csv' CSV HEADER;

-- -----------------------------------------
-- 3. Insert into main buildings table
-- geom is MultiPolygon
-- tiff_path stays NULL
-- -----------------------------------------

INSERT INTO src.buildings (geom, area_m2, confidence, plus_code)
SELECT
    ST_GeomFromText(geom, 4326),
    area_m2,
    confidence,
    plus_code
FROM src._staging_open_buildings
WHERE geom IS NOT NULL;

-- -----------------------------------------
-- 4. Cleanup
-- -----------------------------------------

DROP TABLE src._staging_open_buildings;

COMMIT;

-- -----------------------------------------
-- 5. Report
-- -----------------------------------------

SELECT COUNT(*) AS total_buildings FROM src.buildings;
