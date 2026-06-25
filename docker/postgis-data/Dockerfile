FROM postgis/postgis:18-3.6

LABEL org.opencontainers.image.title="Master Thesis GeoAI PostGIS Data"
LABEL org.opencontainers.image.description="PostGIS database initialized with the GeoAI thesis src and src_google schemas."
LABEL org.opencontainers.image.source="https://github.com/patrickfranzelin/Master-Thesis-GEOAI"

COPY geoai.dump /docker-entrypoint-initdb.d/geoai.dump
COPY restore-geoai.sh /docker-entrypoint-initdb.d/20-restore-geoai.sh
