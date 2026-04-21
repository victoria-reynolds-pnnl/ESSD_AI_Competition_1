"""
model.py — Shared NegBinWrapper class for pickling/unpickling.
Imported by both train.py and evaluate.py.
"""

import numpy as np
import pandas as pd


class NegBinWrapper:
    """
    Serialisable Python handle to the fitted glmmTMB m5.4 model.

    Final model formula:
        n_extreme ~ c_year * extreme_type + as.factor(NERC_ID) + (1 | start_yr)
        family = nbinom2(link = "log")

    Population-level predictions (year random effect set to 0) are computed
    entirely in Python from the stored fixed-effect coefficient table:

        c_year  = start_yr - c_year_mean
        log(mu) = b_Intercept
                + b_c_year * c_year
                + b_extreme_type[E]                (0 if reference level)
                + b_as.factor(NERC_ID)[N]           (0 if reference = NERC1)
                + b_c_year:extreme_type[E] * c_year (0 if reference)
        mu      = exp(log(mu))

    No R installation required at prediction time.
    """

    def __init__(self, rds_path, coef_df, dispersion, c_year_mean,
                 ref_extreme, ref_nerc):
        self.rds_path    = str(rds_path)
        self.coef_df     = coef_df        # index = term, columns include Estimate
        self.dispersion  = dispersion     # NB size parameter (sigma)
        self.c_year_mean = c_year_mean    # centering offset: mean(start_yr) in train+val
        self.ref_extreme = ref_extreme    # reference extreme_type level (coef = 0)
        self.ref_nerc    = ref_nerc       # reference NERC_ID level (coef = 0)
        self.formula     = (
            "n_extreme ~ c_year * extreme_type + "
            "as.factor(NERC_ID) + (1 | start_yr)"
        )
        self.family      = "nbinom2(link='log')"
        self.train_years = (1980, 2018)   # train + val combined

    def predict(self, newdata: pd.DataFrame) -> pd.Series:
        """
        Return population-level predicted event counts for *newdata*.

        Parameters
        ----------
        newdata : pd.DataFrame
            Must contain columns: 'extreme_type', 'NERC_ID', 'start_yr'

        Returns
        -------
        pd.Series of float predicted counts, same index as *newdata*.
        """
        intercept   = float(self.coef_df.loc["(Intercept)", "Estimate"])
        c_year_coef = float(self.coef_df.loc["c_year", "Estimate"])

        def _pred(row):
            c_year = float(row["start_yr"]) - self.c_year_mean
            log_mu = intercept + c_year_coef * c_year

            # extreme_type fixed effect
            et_key = f"extreme_type{row['extreme_type']}"
            if et_key in self.coef_df.index:
                log_mu += float(self.coef_df.loc[et_key, "Estimate"])

            # NERC_ID fixed effect (as.factor)
            nerc_key = f"as.factor(NERC_ID){row['NERC_ID']}"
            if nerc_key in self.coef_df.index:
                log_mu += float(self.coef_df.loc[nerc_key, "Estimate"])

            # c_year:extreme_type interaction
            int_key = f"c_year:extreme_type{row['extreme_type']}"
            if int_key in self.coef_df.index:
                log_mu += float(self.coef_df.loc[int_key, "Estimate"]) * c_year

            return np.exp(log_mu)

        return newdata.apply(_pred, axis=1)
