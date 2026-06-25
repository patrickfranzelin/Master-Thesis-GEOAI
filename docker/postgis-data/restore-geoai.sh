set -e

echo "Initializing GeoAI thesis database from embedded dump"
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-SQL
    CREATE EXTENSION IF NOT EXISTS postgis;
SQL

pg_restore \
    --username "$POSTGRES_USER" \
    --dbname "$POSTGRES_DB" \
    --no-owner \
    --no-acl \
    /docker-entrypoint-initdb.d/geoai.dump

echo "GeoAI thesis database restore complete"
