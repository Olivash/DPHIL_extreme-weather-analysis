"""
Variance decomposition: reforecast intrinsic vs. ERA5 variance,
medium-range (days 0-13) + S2S (days 15-40) extension.

Intrinsic variance isolates the within-ensemble (chaotic/unpredictable)
component of forecast spread by removing the ensemble-mean signal and
adding back daily climatology, then compares it against ERA5's
observed variance over the same valid-date window.

Inputs
------
reforecast_0_t0_13.csv          medium-range reforecast (days 0-13)
variance_plots-S2S.csv          sub-seasonal reforecast (days 15-40)
pnw_box_era5.nc                 ERA5 t2m over the PNW box

Outputs
-------
variance_summary_all.csv        one row per lead day, var_total /
                                 var_intrinsic / var_era5 / ratios
"""

import numpy as np
import pandas as pd
import xarray as xr

# ── Paths ────────────────────────────────────────────────────────────────
ERA5_PATH = "/network/group/aopp/predict/AWH020_AYIM_EXTREME/ERA5/era5_t2m/pnw_box_era5.nc"
S2S_CSV = "/network/group/aopp/predict/MRA001_AYIM_REFORCST/sub_seasonal_full/t2m_prec_0.25/variance_plots-S2S.csv"
MEDIUM_RANGE_CSV = "reforecast_0_t0_13.csv"

OUT_CSV = "variance_summary_all.csv"

# ── Valid-date window (JJA heatwave window) ────────────────────────────────
VALID_START_MMDD = "06-15"
VALID_END_MMDD = "07-20"

MEDIUM_RANGE_LEADS = range(0, 14)
S2S_LEADS = range(15, 41)


def load_era5(path: str) -> pd.DataFrame:
    """Load ERA5 t2m, remove day-of-year climatology, return long-format df."""
    era5 = xr.open_dataset(path)
    #climatology = era5.groupby("time.dayofyear").mean("time")
    #anomalies = era5.groupby("time.dayofyear") - climatology

    era5_df = (
        era5.to_dataframe()
        .reset_index()
        .rename(columns={"time": "date", "t2m": "t2m_era5"})
    )
    era5_df["date"] = pd.to_datetime(era5_df["date"])
    era5_df["valid_mmdd"] = era5_df["date"].dt.strftime("%m-%d")
    return era5_df


def load_medium_range(path: str) -> pd.DataFrame:
    """Load medium-range reforecast csv (already has forecast_date)."""
    df = pd.read_csv(path)
    df["forecast_date"] = pd.to_datetime(df["forecast_date"])
    return df


def load_s2s(path: str) -> pd.DataFrame:
    """Load S2S reforecast csv and derive forecast_date from inidate + step."""
    df = pd.read_csv(path)
    df = df.rename(columns={"time": "hDate", "step": "time", "mx2t6": "t2m"})

    df["time"] = pd.to_timedelta(df["time"])
    df["days"] = df["time"].dt.days
    df["day_month"] = pd.to_datetime(df["inidate"]).dt.strftime("%m-%d")

    df["base_date"] = pd.to_datetime(df["hDate"].astype(str) + "-" + df["day_month"])
    df["forecast_date"] = df["base_date"] + pd.to_timedelta(df["days"], unit="D")
    return df


def attach_era5(reforecast_df: pd.DataFrame, era5_df: pd.DataFrame) -> pd.DataFrame:
    """Merge ERA5 t2m onto reforecast rows by valid (forecast) date."""
    df = reforecast_df.merge(
        era5_df[["date", "t2m_era5"]],
        left_on="forecast_date",
        right_on="date",
        how="left",
    )
    df["valid_mmdd"] = df["forecast_date"].dt.strftime("%m-%d")
    return df


def compute_variance_by_lead(
    reforecast_df: pd.DataFrame,
    era5_df: pd.DataFrame,
    leads: range,
    valid_start_mmdd: str,
    valid_end_mmdd: str,
) -> pd.DataFrame:
    """
    For each lead day: compute intrinsic variance (ensemble spread around
    the ensemble mean, referenced to climatology) and compare against the
    ERA5 variance over the same valid-date window.
    """
    results = []

    for lead in leads:
        df_lead = reforecast_df[reforecast_df["days"] == lead].copy()
        df_lead = df_lead[
            (df_lead["valid_mmdd"] >= valid_start_mmdd)
            & (df_lead["valid_mmdd"] <= valid_end_mmdd)
        ].copy()

        df_lead["ensemble_mean"] = df_lead.groupby(["inidate", "hDate", "days"])[
            "t2m"
        ].transform("mean")
        df_lead["climatology"] = df_lead.groupby("day_month")["t2m"].transform("mean")
        df_lead["intrinsic"] = (
            df_lead["t2m"] - df_lead["ensemble_mean"] + df_lead["climatology"]
        )

        valid_days = np.sort(df_lead["valid_mmdd"].astype(str).unique())
        era5_selected = era5_df.loc[
            era5_df["valid_mmdd"].astype(str).isin(valid_days)
        ].copy()

        var_total = df_lead["t2m"].var()
        var_intrinsic = df_lead["intrinsic"].var()
        var_era5 = era5_selected["t2m_era5"].var()

        results.append(
            {
                "days": lead,
                "n_obs": len(df_lead),
                "var_total": var_total,
                "var_intrinsic": var_intrinsic,
                "var_era5": var_era5,
                "ratio_intrinsic_total": var_intrinsic / var_total,
                "ratio_total_era5": var_total / var_era5,
                "ratio_intrinsic_era5": var_intrinsic / var_era5,
            }
        )

    return pd.DataFrame(results).set_index("days")


def main():
    era5_df = load_era5(ERA5_PATH)

    medium_range = attach_era5(load_medium_range(MEDIUM_RANGE_CSV), era5_df)
    s2s = attach_era5(load_s2s(S2S_CSV), era5_df)

    variance_medium_range = compute_variance_by_lead(
        medium_range, era5_df, MEDIUM_RANGE_LEADS, VALID_START_MMDD, VALID_END_MMDD
    )
    variance_s2s = compute_variance_by_lead(
        s2s, era5_df, S2S_LEADS, VALID_START_MMDD, VALID_END_MMDD
    )

    variance_summary_all = (
        pd.concat([variance_medium_range, variance_s2s])
        .sort_index()
        .dropna(subset=["var_total"])
    )

    variance_summary_all.to_csv(OUT_CSV)
    print(variance_summary_all)
    print(f"\nSaved -> {OUT_CSV}")


if __name__ == "__main__":
    main()