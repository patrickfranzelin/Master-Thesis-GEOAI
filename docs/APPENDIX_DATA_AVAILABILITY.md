# Appendix: Data Availability

The thesis data used by this code repository is available as a Dockerized PostGIS data image on GitHub Container Registry. The raw database dump is not committed to git; it is embedded into the container image and restored on first startup.

## Public Docker Image

```text
image: ghcr.io/patrickfranzelin/master-thesis-geoai-data:latest
```

The image must be public in GitHub Packages for anonymous users to pull it. If Docker returns an authorization error, open the package settings in GitHub and change package visibility to public.

Run with Docker Compose:

```powershell
docker compose up -d data-db
$env:PG_CONN="postgresql://postgres:geoai@localhost:5434/geoai"
```

Run directly:

```powershell
docker run --name geoai-postgis-data -p 5434:5432 `
  -e POSTGRES_DB=geoai `
  -e POSTGRES_USER=postgres `
  -e POSTGRES_PASSWORD=geoai `
  ghcr.io/patrickfranzelin/master-thesis-geoai-data:latest
```

## Database Access

After starting the image, access the data with:

```text
container: geoai-postgis-data
host: localhost
port: 5434
database: geoai
user: postgres
password: geoai
connection string: postgresql://postgres:geoai@localhost:5434/geoai
```

If port `5434` is already used on another machine, choose another host port with `-TargetPort` and update the connection string accordingly.

## Local Copy Workflow

If you maintain a local PostgreSQL source database and want to rebuild the data image, first create or refresh the Docker dump:

```powershell
.\scripts\db_copy_local_to_docker.ps1 -SourcePassword "<local-db-password>" -TargetPort 5434
$env:PG_CONN="postgresql://postgres:geoai@localhost:5434/geoai"
```

The copy workflow keeps the original local database unchanged and restores a dump into the Docker volume.

The image build context is under `docker/postgis-data`. Copy the verified dump into that directory before building:

```powershell
Copy-Item .docker_db_backups\geoai-20260625_114034.dump docker\postgis-data\geoai.dump
docker build -t ghcr.io/patrickfranzelin/master-thesis-geoai-data:latest docker/postgis-data
```

## Available Schemas

| Schema | Description |
| --- | --- |
| `src` | Smaller development schema. |
| `src_google` | Main thesis-scale Google Open Buildings experiment and evaluation schema. |

## Main Thesis Tables

| Table | Rows in Docker copy | Description |
| --- | ---: | --- |
| `src_google.aoi` | 7 | Study area geometries. |
| `src_google.buildings` | 8,269 | Input Google Open Buildings footprints with TIFF links. |
| `src_google.building_mlqa` | 3,711 | Multimodal quality assessment outputs and prompt points. |
| `src_google.detected_house` | 8,608 | SAM and global-discovery house detections. |
| `src_google.detected_house_regularized` | 6,767 | Postprocessed and regularized building footprints. |
| `src_google.detected_tree` | 8,490 | Tree/occlusion detections used during postprocessing. |
| `src_google.evaluation` | 1,263 | Manual building-level evaluation records. |
| `src_google.semantic_description_evaluation` | 102 | Manual evaluation of generated error descriptions. |
| `src_google.tiffs` | 1 | Registered raster extent metadata. |

All main geometry columns use EPSG:4326. Building inputs are stored as `MULTIPOLYGON`; detected and regularized outputs are stored as `POLYGON`.

## Backup Artifact

The data image was built from the latest verified local Docker import:

```text
.docker_db_backups/geoai-20260625_114034.dump
```

This dump and `docker/postgis-data/geoai.dump` are intentionally ignored by git. They can be recreated from the local PostgreSQL source database with the copy script above.
