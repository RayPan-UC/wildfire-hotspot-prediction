# download raw data

# 1. Alberta boundary
# https://www12.statcan.gc.ca/census-recensement/2011/geo/bound-limit/files-fichiers/lcsd000a25p_e.zip
# save to: wildfire-hotspot-prediction\data_raw\AOI_boundary

# 2. Fort McMurray
"""
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {},
      "geometry": {
        "coordinates": [
          [
            [
              -112.63443087044945,
              57.38003219306427
            ],
            [
              -112.63443087044945,
              56.15711918602531
            ],
            [
              -110.00197057703087,
              56.15711918602531
            ],
            [
              -110.00197057703087,
              57.38003219306427
            ],
            [
              -112.63443087044945,
              57.38003219306427
            ]
          ]
        ],
        "type": "Polygon"
      }
    }
  ]
}
"""
# save to: wildfire-hotspot-prediction\data_raw\AOI_boundary


# 3. weather data (historical)
import cdsapi

client = cdsapi.Client()

dataset = 'reanalysis-era5-pressure-levels'
request = {
  'product_type': ['reanalysis'],
  'variable': ['geopotential'],
  'year': ['2024'],
  'month': ['03'],
  'day': ['01'],
  'time': ['13:00'],
  'pressure_level': ['1000'],
  'data_format': 'grib',
}
target = 'download.grib'

client.retrieve(dataset, request, target)
#save to: wildfire-hotspot-prediction\data_raw\weather\ERA5

# 4. weather data (prediction)
"""
High Resolution Deterministic Prediction System (HRDPS) data
The files have the following nomenclature :

{YYYYMMDD}T{HH}Z_MSC_HRDPS_{VAR}_{LVLTYPE-LVL}_{Grid}{resolution}_PT{hhh}H.grib2
{YYYYMMDD}T{HH}Z_MSC_HRDPS-WEonG_{VAR}{LVLTYPE-LVL}{Grid}{resolution}_PT{hhh}H.grib2
where :

YYYYMMDD : Year, month and day of the beginning of the forecast
T : Time delimiter according to ISO8601 norms
HH : UTC run time [00, 06, 12, 18]
Z : Time zone (UTC hour)
MSC : Constant string indicating the Meteorological Service of Canada, source of data
HRDPS : Constant string indicating that the data is from the High Resolution Deterministic Prediction System
HRDPS-WEonG : Constant string indicating that the data is from the weather elements on the grid of the High Resolution Deterministic Prediction System
VAR : Variable type included in the file (ex: UGRD)
LVLTYPE-LVL : Vertical level type and level value [ex: Sfc for surface, EATM for the entire atmospheric column, DBS-10-20cm layer between 10 and 20cm under surface, AGL-10m for 10m above ground level]
Grid : Horizontal grid [RLatLon]
resolution : 0.0225. Indicating resolution in degree [0.0225°(environ 2.5km)] in latitude and longitude directions
PT{hhh}H : Forecast hours based on ISO8601 norms. P, T and H are constant character designating Period, Time and Hour. "hhh" is the forecast hour [000, 001, 002, ..., 048]
grib2 : Constant string indicating the GRIB2 format is used
"""

# 5. SRTM
# use google earth engine

# 6. fuel type
# historical: https://cwfis.cfs.nrcan.gc.ca/downloads/fuels/archive/National_FBP_Fueltypes_version2014b.zip
# current: https://cwfis.cfs.nrcan.gc.ca/downloads/fuels/current/FBP_fueltypes_Canada_30m_EPSG3978_20240522.zip

# 7. hotspot
# historical: https://cwfis.cfs.nrcan.gc.ca/downloads/hotspots/archive/2016_hotspots.zip
# real-time: ???