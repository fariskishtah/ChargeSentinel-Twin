"""
Feature Engineering for Solar Forecasting (New Cairo Renewable Pipeline).

Guarantees 100% Leakage-Safe Feature Generation:
Every feature for predicting horizon h at target timestamp (T_0 + h)
is derived strictly from information available at or before forecast origin T_0:
- Calendar & Astronomical solar geometry for target timestamp (T_0 + h)
- NWP Weather Forecasts (_previous_day1 / _previous_day2 runs available at T_0)
- Historical PV power observations (lags and rolling statistics strictly <= T_0)
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

# New Cairo Coordinates and System Constants
NEW_CAIRO_LAT = 30.048813
NEW_CAIRO_LON = 31.523438
SOLAR_CONSTANT = 1367.0  # W/m^2
PV_SYSTEM_CAPACITY_KW = 100.0  # 100 kWp system

# Feature Column Names
CALENDAR_FEATURES = [
    "hour",
    "month",
    "day_of_year",
    "day_of_week",
    "sin_hour",
    "cos_hour",
    "sin_day_of_year",
    "cos_day_of_year",
    "sin_month",
    "cos_month",
]

SOLAR_GEO_FEATURES = [
    "solar_elevation_deg",
    "solar_zenith_deg",
    "solar_extraterrestrial_ghi",
    "is_daylight",
]

NWP_WEATHER_FEATURES = [
    "temperature_2m_previous_day1",
    "temperature_2m_previous_day2",
    "cloud_cover_previous_day1",
    "cloud_cover_previous_day2",
    "direct_normal_irradiance_previous_day1",
    "direct_normal_irradiance_previous_day2",
    "shortwave_radiation_previous_day1",
    "shortwave_radiation_previous_day2",
    "wind_speed_10m_previous_day1",
    "wind_speed_10m_previous_day2",
]

INTERACTION_FEATURES = [
    "horizon",
    "temp_diff_runs",
    "cloud_diff_runs",
    "rad_diff_runs",
    "dni_diff_runs",
    "wind_diff_runs",
    "ghi_to_extra_ratio",
    "dni_to_ghi_ratio",
    "estimated_cell_temp",
]

PV_LAG_FEATURES = [
    "pv_origin_power",
    "pv_origin_lag1",
    "pv_origin_lag2",
    "pv_mean_24h_origin",
    "pv_max_24h_origin",
    "pv_std_24h_origin",
    "pv_mean_48h_origin",
    "pv_max_48h_origin",
    "pv_same_hour_prev_day",
    "pv_same_hour_prev_day2",
    "pv_seasonal_persistence",
]

ALL_FEATURE_COLUMNS = (
    CALENDAR_FEATURES
    + SOLAR_GEO_FEATURES
    + NWP_WEATHER_FEATURES
    + INTERACTION_FEATURES
    + PV_LAG_FEATURES
)


def calculate_solar_geometry(
    timestamps: Union[pd.DatetimeIndex, List[pd.Timestamp]],
    lat: float = NEW_CAIRO_LAT,
    lon: float = NEW_CAIRO_LON,
) -> pd.DataFrame:
    """
    Calculate high-precision solar geometry for given timestamps in UTC.

    Computes:
    - Solar declination (Cooper equation)
    - Equation of Time (EoT)
    - Solar Time & Solar Hour Angle (omega)
    - Solar Elevation Angle (alpha) & Solar Zenith Angle (theta_z)
    - Extraterrestrial Global Horizontal Irradiance (G_0)
    - Binary daylight indicator (alpha > 0)
    """
    ts = pd.to_datetime(timestamps)
    day_of_year = ts.dayofyear.to_numpy()
    hour_utc = (ts.hour + ts.minute / 60.0 + ts.second / 3600.0).to_numpy()

    # Solar declination delta in radians
    gamma = 2.0 * np.pi * (day_of_year - 1) / 365.0
    declination = 23.45 * np.sin(np.radians((360.0 / 365.0) * (day_of_year + 284)))
    decl_rad = np.radians(declination)

    # Equation of Time (minutes)
    eot = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2.0 * gamma)
        - 0.040849 * np.sin(2.0 * gamma)
    )

    # Solar Time in hours
    time_offset = eot + 4.0 * lon  # 4 min per degree longitude
    solar_time = np.mod(hour_utc + time_offset / 60.0, 24.0)

    # Hour angle omega (-180 to +180 deg)
    hour_angle = 15.0 * (solar_time - 12.0)
    ha_rad = np.radians(hour_angle)

    lat_rad = np.radians(lat)

    # Solar elevation angle
    sin_alpha = np.sin(lat_rad) * np.sin(decl_rad) + np.cos(lat_rad) * np.cos(
        decl_rad
    ) * np.cos(ha_rad)
    sin_alpha = np.clip(sin_alpha, -1.0, 1.0)
    elevation_deg = np.degrees(np.arcsin(sin_alpha))
    zenith_deg = 90.0 - elevation_deg

    # Extraterrestrial irradiance (W/m^2) on horizontal plane
    earth_sun_dist_corr = 1.0 + 0.033 * np.cos(2.0 * np.pi * day_of_year / 365.0)
    ghi_extra = np.maximum(0.0, SOLAR_CONSTANT * earth_sun_dist_corr * sin_alpha)

    # Daylight indicator (elevation > 0 deg)
    is_daylight = (elevation_deg > 0.0).astype(float)

    return pd.DataFrame(
        {
            "solar_elevation_deg": elevation_deg,
            "solar_zenith_deg": zenith_deg,
            "solar_extraterrestrial_ghi": ghi_extra,
            "is_daylight": is_daylight,
        },
        index=ts,
    )


def extract_calendar_features(timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    """Extract cyclical and categorical calendar features."""
    ts = pd.to_datetime(timestamps)
    hour = ts.hour.to_numpy()
    month = ts.month.to_numpy()
    day_of_year = ts.dayofyear.to_numpy()
    day_of_week = ts.dayofweek.to_numpy()

    return pd.DataFrame(
        {
            "hour": hour,
            "month": month,
            "day_of_year": day_of_year,
            "day_of_week": day_of_week,
            "sin_hour": np.sin(2.0 * np.pi * hour / 24.0),
            "cos_hour": np.cos(2.0 * np.pi * hour / 24.0),
            "sin_day_of_year": np.sin(2.0 * np.pi * day_of_year / 365.0),
            "cos_day_of_year": np.cos(2.0 * np.pi * day_of_year / 365.0),
            "sin_month": np.sin(2.0 * np.pi * month / 12.0),
            "cos_month": np.cos(2.0 * np.pi * month / 12.0),
        },
        index=ts,
    )


class SolarFeaturePipeline:
    """
    Leakage-safe feature extraction pipeline for 24h/48h solar power forecasting.
    Works seamlessly in batch training and live single-payload inference.
    """

    def __init__(
        self,
        lat: float = NEW_CAIRO_LAT,
        lon: float = NEW_CAIRO_LON,
        system_capacity_kw: float = PV_SYSTEM_CAPACITY_KW,
    ):
        self.lat = lat
        self.lon = lon
        self.system_capacity_kw = system_capacity_kw
        self.feature_columns = ALL_FEATURE_COLUMNS

    def build_forecast_features(
        self,
        forecast_origin: pd.Timestamp,
        horizon: int,
        weather_df: pd.DataFrame,
        historical_power_series: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Build feature matrix for a specific forecast origin and horizon (24 or 48 h).

        Parameters:
        -----------
        forecast_origin: pd.Timestamp
            The origin timestamp T_0.
        horizon: int
            Number of hours ahead (24 or 48).
        weather_df: pd.DataFrame
            Weather forecast indexed by target timestamps (T_0 + 1h .. T_0 + horizon*h),
            containing _previous_day1 and _previous_day2 columns.
        historical_power_series: Optional[pd.Series]
            Hourly PV power strictly <= forecast_origin, in kW.
            If None, safe defaults / zero-lags are used.

        Returns:
        --------
        pd.DataFrame: Feature matrix of shape (horizon, n_features) indexed by target timestamp.
        """
        forecast_origin = pd.to_datetime(forecast_origin, utc=True)
        target_times = pd.date_range(
            start=forecast_origin + pd.Timedelta(hours=1),
            periods=horizon,
            freq="h",
        )

        # 1. Calendar features for target timestamps
        df_cal = extract_calendar_features(target_times)

        # 2. Solar geometry for target timestamps
        df_geo = calculate_solar_geometry(target_times, lat=self.lat, lon=self.lon)

        # 3. NWP Weather features
        weather_aligned = weather_df.copy()
        if not isinstance(weather_aligned.index, pd.DatetimeIndex):
            if "time" in weather_aligned.columns:
                weather_aligned["time"] = pd.to_datetime(weather_aligned["time"])
                weather_aligned = weather_aligned.set_index("time")
            elif "timestamp" in weather_aligned.columns:
                weather_aligned["timestamp"] = pd.to_datetime(weather_aligned["timestamp"])
                weather_aligned = weather_aligned.set_index("timestamp")

        # Normalize every timestamp to UTC before alignment. Naive source timestamps
        # are interpreted as UTC (the source data is documented in UTC).
        weather_aligned.index = pd.to_datetime(weather_aligned.index, utc=True)
        weather_aligned = weather_aligned[~weather_aligned.index.duplicated(keep="last")].sort_index()

        # Align with target times
        weather_sub = weather_aligned.reindex(target_times)

        # The previous-day-1 NWP run is not treated as available after lead 24h.
        # For leads 25-48, use the older previous-day-2 run as the primary run.
        if horizon > 24:
            beyond_24h = np.arange(1, horizon + 1) > 24
            for base in (
                "temperature_2m", "cloud_cover", "direct_normal_irradiance",
                "shortwave_radiation", "wind_speed_10m",
            ):
                p1, p2 = f"{base}_previous_day1", f"{base}_previous_day2"
                if p1 in weather_sub.columns and p2 in weather_sub.columns:
                    weather_sub.loc[target_times[beyond_24h], p1] = weather_sub.loc[target_times[beyond_24h], p2]

        # For 48h forecasts, leads 25-48 must have the older NWP run available.
        # Never substitute previous_day1 for missing previous_day2 in that range.
        if horizon > 24:
            late_times = target_times[24:]
            required_day2 = [c for c in NWP_WEATHER_FEATURES if c.endswith("_previous_day2")]
            missing_cols = [c for c in required_day2 if c not in weather_sub.columns]
            if missing_cols:
                raise ValueError(f"48h forecast requires previous_day2 weather fields: {missing_cols}")
            missing_late = weather_sub.loc[late_times, required_day2].isna().any()
            if missing_late.any():
                bad = missing_late[missing_late].index.tolist()
                raise ValueError(f"48h forecast has unavailable previous_day2 values for leads 25-48: {bad}")

        # Ensure all required NWP columns exist. Missing optional day-2 fields may
        # fall back only for 24h forecasts, where they are not required.
        for col in NWP_WEATHER_FEATURES:
            if col not in weather_sub.columns:
                fallback = col.replace("_previous_day2", "_previous_day1")
                if horizon == 24 and col.endswith("_previous_day2") and fallback in weather_sub.columns:
                    weather_sub[col] = weather_sub[fallback]
                else:
                    weather_sub[col] = np.nan

        # Fill missing weather fields within the same target timestamp only.
        # Never bfill/ffill across rows: that can copy another lead time's value.
        for base in (
            "temperature_2m", "cloud_cover", "direct_normal_irradiance",
            "shortwave_radiation", "wind_speed_10m",
        ):
            p1, p2 = f"{base}_previous_day1", f"{base}_previous_day2"
            if p1 in weather_sub.columns and p2 in weather_sub.columns:
                weather_sub[p1] = weather_sub[p1].fillna(weather_sub[p2])
                weather_sub[p2] = weather_sub[p2].fillna(weather_sub[p1])
        weather_sub = weather_sub.fillna(0.0)

        # 4. Interaction / NWP Derived features
        horizons = np.arange(1, horizon + 1)
        temp_diff = (
            weather_sub["temperature_2m_previous_day1"]
            - weather_sub["temperature_2m_previous_day2"]
        )
        cloud_diff = (
            weather_sub["cloud_cover_previous_day1"]
            - weather_sub["cloud_cover_previous_day2"]
        )
        rad_diff = (
            weather_sub["shortwave_radiation_previous_day1"]
            - weather_sub["shortwave_radiation_previous_day2"]
        )
        dni_diff = (
            weather_sub["direct_normal_irradiance_previous_day1"]
            - weather_sub["direct_normal_irradiance_previous_day2"]
        )
        wind_diff = (
            weather_sub["wind_speed_10m_previous_day1"]
            - weather_sub["wind_speed_10m_previous_day2"]
        )

        ghi_p1 = weather_sub["shortwave_radiation_previous_day1"]
        extra_ghi = df_geo["solar_extraterrestrial_ghi"]
        ghi_to_extra = np.clip(ghi_p1 / (extra_ghi + 10.0), 0.0, 1.5)

        dni_p1 = weather_sub["direct_normal_irradiance_previous_day1"]
        dni_to_ghi = np.clip(dni_p1 / (ghi_p1 + 10.0), 0.0, 1.0)

        # Approximate PV cell temperature: T_cell = T_amb + (NOCT - 20)/800 * GHI
        t_amb = weather_sub["temperature_2m_previous_day1"]
        cell_temp = t_amb + ((45.0 - 20.0) / 800.0) * ghi_p1

        df_interaction = pd.DataFrame(
            {
                "horizon": horizons,
                "temp_diff_runs": temp_diff,
                "cloud_diff_runs": cloud_diff,
                "rad_diff_runs": rad_diff,
                "dni_diff_runs": dni_diff,
                "wind_diff_runs": wind_diff,
                "ghi_to_extra_ratio": ghi_to_extra,
                "dni_to_ghi_ratio": dni_to_ghi,
                "estimated_cell_temp": cell_temp,
            },
            index=target_times,
        )

        # 5. Leakage-safe PV Historical Power Features
        pv_lags = self._compute_safe_pv_features(
            forecast_origin=forecast_origin,
            target_times=target_times,
            horizons=horizons,
            historical_power=historical_power_series,
        )

        # Combine all features
        features = pd.concat(
            [df_cal, df_geo, weather_sub[NWP_WEATHER_FEATURES], df_interaction, pv_lags],
            axis=1,
        )

        return features[self.feature_columns]

    def _compute_safe_pv_features(
        self,
        forecast_origin: pd.Timestamp,
        target_times: pd.DatetimeIndex,
        horizons: np.ndarray,
        historical_power: Optional[pd.Series],
    ) -> pd.DataFrame:
        """
        Compute historical PV lags and rolling features strictly <= forecast_origin.
        """
        n = len(target_times)
        if historical_power is None or len(historical_power) == 0:
            # Safe defaults if no history provided (e.g. cold start)
            return pd.DataFrame(
                {col: np.zeros(n, dtype=float) for col in PV_LAG_FEATURES},
                index=target_times,
            )

        # Ensure DatetimeIndex and sort
        hist = historical_power.copy()
        hist.index = pd.to_datetime(hist.index, utc=True)
        hist = hist[~hist.index.duplicated(keep="last")].sort_index()

        # Strict cutoff: power strictly <= forecast_origin
        hist = hist.loc[:forecast_origin]

        # Origin power observations
        p_origin = float(hist.loc[forecast_origin]) if forecast_origin in hist.index else (
            float(hist.iloc[-1]) if len(hist) > 0 else 0.0
        )

        t_minus_1 = forecast_origin - pd.Timedelta(hours=1)
        p_origin_lag1 = float(hist.loc[t_minus_1]) if t_minus_1 in hist.index else p_origin

        t_minus_2 = forecast_origin - pd.Timedelta(hours=2)
        p_origin_lag2 = float(hist.loc[t_minus_2]) if t_minus_2 in hist.index else p_origin_lag1

        # Rolling statistics over last 24h & 48h prior to origin
        w24 = hist.loc[forecast_origin - pd.Timedelta(hours=23) : forecast_origin]
        mean_24h = float(w24.mean()) if len(w24) > 0 else 0.0
        max_24h = float(w24.max()) if len(w24) > 0 else 0.0
        std_24h = float(w24.std()) if len(w24) > 1 else 0.0

        w48 = hist.loc[forecast_origin - pd.Timedelta(hours=47) : forecast_origin]
        mean_48h = float(w48.mean()) if len(w48) > 0 else mean_24h
        max_48h = float(w48.max()) if len(w48) > 0 else max_24h

        # Same-hour historical values for target times (T_0 + h)
        same_hour_p1 = np.zeros(n, dtype=float)
        same_hour_p2 = np.zeros(n, dtype=float)
        persistence = np.zeros(n, dtype=float)

        for i, (t_target, h) in enumerate(zip(target_times, horizons)):
            # Same hour 24h prior to target time: t_target - 24h
            t_lag24 = t_target - pd.Timedelta(hours=24)
            # Same hour 48h prior to target time: t_target - 48h
            t_lag48 = t_target - pd.Timedelta(hours=48)

            val_lag24 = float(hist.loc[t_lag24]) if t_lag24 in hist.index else 0.0
            val_lag48 = float(hist.loc[t_lag48]) if t_lag48 in hist.index else val_lag24

            same_hour_p2[i] = val_lag48

            # If horizon h <= 24, t_target - 24h <= forecast_origin (known at origin!)
            # If horizon h > 24, t_target - 24h is in the FUTURE (> forecast_origin).
            # To PREVENT LEAKAGE, for h > 24 we use t_target - 48h (which is <= origin)!
            if h <= 24:
                same_hour_p1[i] = val_lag24
                persistence[i] = val_lag24
            else:
                same_hour_p1[i] = val_lag48
                persistence[i] = val_lag48

        return pd.DataFrame(
            {
                "pv_origin_power": np.full(n, p_origin),
                "pv_origin_lag1": np.full(n, p_origin_lag1),
                "pv_origin_lag2": np.full(n, p_origin_lag2),
                "pv_mean_24h_origin": np.full(n, mean_24h),
                "pv_max_24h_origin": np.full(n, max_24h),
                "pv_std_24h_origin": np.full(n, std_24h),
                "pv_mean_48h_origin": np.full(n, mean_48h),
                "pv_max_48h_origin": np.full(n, max_48h),
                "pv_same_hour_prev_day": same_hour_p1,
                "pv_same_hour_prev_day2": same_hour_p2,
                "pv_seasonal_persistence": persistence,
            },
            index=target_times,
        )

    def build_dataset_from_raw(
        self,
        df_weather: pd.DataFrame,
        pv_power_kw: pd.Series,
        horizon: int = 24,
        origin_step_hours: int = 6,
        start_date: Optional[str] = "2024-01-22 00:00:00",
        end_date: Optional[str] = "2024-06-30 23:00:00",
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """
        Build rolling-origin training/evaluation tabular dataset.

        Parameters:
        -----------
        df_weather: pd.DataFrame
            Weather DataFrame indexed by time.
        pv_power_kw: pd.Series
            PV power series in kW indexed by time.
        horizon: int
            24 or 48 hours.
        origin_step_hours: int
            Step size in hours between successive forecast origins (e.g. 6 or 24).
        start_date, end_date: Optional[str]
            Date boundaries for forecast origins.

        Returns:
        --------
        (X, y, metadata_df):
            X: Feature matrix
            y: Target power (kW)
            metadata_df: DataFrame with 'origin', 'target_time', 'horizon'
        """
        weather = df_weather.copy()
        if "time" in weather.columns and not isinstance(weather.index, pd.DatetimeIndex):
            weather["time"] = pd.to_datetime(weather["time"])
            weather = weather.set_index("time")

        weather.index = pd.to_datetime(weather.index, utc=True)
        weather = weather[~weather.index.duplicated(keep="last")].sort_index()

        pv = pv_power_kw.copy()
        if not isinstance(pv.index, pd.DatetimeIndex):
            pv.index = weather.index
        pv.index = pd.to_datetime(pv.index, utc=True)
        pv = pv[~pv.index.duplicated(keep="last")].sort_index()

        # Interpolate small missing values in target
        pv = pv.interpolate(method="time").fillna(0.0)

        # Determine eligible origins
        all_times = weather.index
        min_origin = pd.to_datetime(start_date, utc=True) if start_date else all_times[48]
        max_origin = (
            pd.to_datetime(end_date, utc=True) - pd.Timedelta(hours=horizon)
            if end_date
            else all_times[-horizon - 1]
        )

        origin_candidates = all_times[
            (all_times >= min_origin) & (all_times <= max_origin)
        ]
        origins = origin_candidates[::origin_step_hours]

        X_list = []
        y_list = []
        meta_list = []

        for origin in origins:
            target_times = pd.date_range(
                start=origin + pd.Timedelta(hours=1),
                periods=horizon,
                freq="h",
            )
            if target_times[-1] > all_times[-1]:
                continue

            # Build features for this origin
            feat = self.build_forecast_features(
                forecast_origin=origin,
                horizon=horizon,
                weather_df=weather,
                historical_power_series=pv.loc[:origin],
            )

            # Targets
            targets = pv.reindex(target_times).values

            meta = pd.DataFrame(
                {
                    "origin": origin,
                    "target_time": target_times,
                    "horizon": np.arange(1, horizon + 1),
                },
                index=target_times,
            )

            X_list.append(feat)
            y_list.append(pd.Series(targets, index=target_times))
            meta_list.append(meta)

        X = pd.concat(X_list, axis=0).reset_index(drop=True)
        y = pd.concat(y_list, axis=0).reset_index(drop=True)
        meta_df = pd.concat(meta_list, axis=0).reset_index(drop=True)

        return X, y, meta_df


def verify_no_leakage(
    meta_df: pd.DataFrame,
    X: pd.DataFrame,
) -> bool:
    """
    Sanity check function asserting that:
    1. Every target_time > origin
    2. Target_time - origin == horizon in hours
    3. No NaNs in feature matrix
    """
    assert len(X) == len(meta_df), "Length mismatch between X and metadata."
    origins = pd.to_datetime(meta_df["origin"])
    target_times = pd.to_datetime(meta_df["target_time"])
    horizons = meta_df["horizon"]

    diff_hours = (target_times - origins).dt.total_seconds() / 3600.0
    assert (diff_hours == horizons).all(), "Horizon mismatch detected!"
    assert (diff_hours > 0).all(), "Target time is not strictly after origin!"
    assert X.isnull().sum().sum() == 0, f"NaNs found in feature matrix: {X.isnull().sum()}"
    return True
