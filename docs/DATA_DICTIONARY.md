# Data dictionary

`Yes` under inference means available to the corresponding historical inference contract; `No` means excluded. Jiaxing timestamps are naive source operational time; their timezone is unverified. PVGIS and Open-Meteo describe modelled/reference solar data, not measured station generation.

| Field | Source / type | Meaning / unit | Role; training | Inference | Leakage note |
|---|---|---|---|---|---|
| `Charging Post ID` | Jiaxing / integer | Charging post identifier | Session-risk charger feature; yes | Yes, as `charger_id` | Identifier may encode site history; no outcome content. |
| `Location Information` | Jiaxing / text | Source location label | Session-risk feature; yes | Yes, as `location` | Source geography, not Egyptian location. |
| `District Name` | Jiaxing / text | Source district label | Session-risk feature; yes | Yes, as `district` | Source geography only. |
| `Start Time` | Jiaxing / datetime text | Session start, source operational time | Session-risk hour/day/month/weekend; yes. Demand arrival aggregation; yes | Yes at session start | Later timestamps excluded. |
| `End Time` | Jiaxing / datetime text | Session completion time | Demand energy aggregation only; yes | No for session risk or demand forecast | Would leak completion at session start. |
| `Transaction power/kwh` | Jiaxing / float | Completed session energy despite source name / kWh | Demand target and hourly aggregation; yes | No for session-start risk | Final energy is post-event. |
| `end_cause` | Jiaxing / text | Recorded session end cause | Source of retrospective risk target; target construction only | No | Direct outcome leakage. |
| `is_abnormal` | Jiaxing / integer | Derived abnormal end-cause flag | Session-risk target only | No | Never a model input. |
| `is_full_charge`, `is_user_stop`, `is_nonsystem_fault`, `is_system_issue`, `is_EV_fault`, `is_other_fault` | Jiaxing / integer | Derived end-cause indicators | Audit/target context only | No | Outcome-derived. |
| `Payment time` | Jiaxing / datetime text | Payment timestamp | No current model input | No | Post-event; old PR used it. |
| `Electricity cost/Yuan`, `Service charge/Yuan`, `Transaction Amount/Yuan`, `Actual Payment/Yuan` | Jiaxing / float | Recorded monetary transaction fields / Yuan | No current model input | No | Final financial values unavailable at session start. |
| `timestamp` | Derived Jiaxing hourly / datetime text | Beginning of source operational hour | Demand and grid-load index; yes | Yes, historical hour | No timezone conversion asserted. |
| `energy_kwh` | Derived Jiaxing hourly / float | Completed-session energy credited at End Time hour / kWh | Demand forecast target and grid-load observed signal; yes | Observed only after hour | Proxy, not instantaneous kW or meter reading. |
| `sessions` | Derived Jiaxing hourly / integer | Sessions starting in hour / count | Demand forecast target and grid-load observed signal; yes | Observed after hour | Different event timing from `energy_kwh`. |
| `power_kw` | PVGIS H1 runtime / float | Modelled 100 kWp reference PV output / kW | Solar target and historical lags; yes | Only past observations | HH:09 source reading is not available at HH:00 origin. |
| `timestamp` | PVGIS H1 runtime / datetime text | Hourly alignment label | Solar target/history index; yes | Past labels only | Ordinally aligned to Open-Meteo UTC labels. |
| `timestamp` | Open-Meteo H1 runtime / datetime text | Target/forecast hour label | Solar NWP lookup; yes | Future target hours with already available prior run | Label is not issue time. |
| `temperature_2m_previous_day1/2` | Open-Meteo / float | Prior-run air temperature / °C | Solar features; yes | Yes, from available prior run | Day2 used at longer lead when day1 is unavailable. |
| `cloud_cover_previous_day1/2` | Open-Meteo / float | Prior-run cloud cover / source units | Solar features; yes | Yes, prior run only | No realized future weather. |
| `direct_normal_irradiance_previous_day1/2` | Open-Meteo / float | Prior-run direct normal irradiance / source units | Solar features; yes | Yes, prior run only | No realized future irradiance. |
| `shortwave_radiation_previous_day1/2` | Open-Meteo / float | Prior-run shortwave radiation / source units | Solar features; yes | Yes, prior run only | No realized future radiation. |
| `wind_speed_10m_previous_day1/2` | Open-Meteo / float | Prior-run 10m wind speed / source units | Solar features; yes | Yes, prior run only | No realized future wind. |

Units for Open-Meteo fields are inherited from the raw provider response and are not independently re-labelled in the committed runtime CSV. See `team/mazen_solar/integration/source_features.py` for exact feature transformations and `team/waseem_demand/demand/features.py` for causal demand lags.
