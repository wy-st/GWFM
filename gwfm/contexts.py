"""Attribute names used by physical output constraints; no downstream masking."""

MODALITIES = ("climate", "pollution", "sociodemographic")
LOG_FEATURES = {
    "annual_precipitation_mm",
    "pm25_ug_m3",
    "no2_ug_m3",
    "ozone_ug_m3",
    "emission_index",
    "population_density_km2",
}
LOGIT_FEATURES = {
    "relative_humidity",
    "hot_day_fraction",
    "deprivation_index",
    "older_population_share",
    "healthcare_access",
}
