# Predeclared event and forecast target

**Reference:** 1991–2010, entirely earlier than every supervised split. This is a 20-year pilot reference, not a standard 30-year climate normal. Each calendar-day threshold pools a circular ±15-day seasonal window. Leap day is retained in the daily sequence and shares 28 February's reference-day index; March dates stay aligned. Exact reference sample counts are exported.

**Heat:** daily maximum 2 m air temperature strictly exceeds both its local seasonal 90th percentile and 30°C, for at least three consecutive UTC days ending today. A heat state is recognised on the third and subsequent hot day. The earlier two days are not retroactively relabelled using future information.

**Drought proxy:** trailing 90-day precipitation total, including today, is strictly below the local seasonal 20th percentile of the same trailing total in the reference period. This is a meteorological precipitation-deficit proxy. It is **not SPEI, SPI, soil-moisture drought, agricultural damage or water scarcity**. Zero percentile thresholds can yield no strict exceedances in very dry seasons; zero thresholds are not silently changed.

**Compound cell-day:** both heat and drought proxy states are true on that day. The alternate definition uses the seasonal 10th percentile for the drought proxy, retaining the heat definition. Its label agreement and fixed-model sensitivity are reported; it is not used to choose favourable results.

**Forecast:** the label on day t+7 is predicted from 56 antecedent daily feature vectors ending on day t. Each vector includes a trailing-90-day rainfall feature; the complete raw dependency therefore extends to t−144. All partitions enforce that the entire raw dependency and the target are within the same declared temporal period. The forecast is retrospective in meteorological valid-time; reanalysis publication latency is not simulated, so operational seven-day forecasting skill is not established.

**Input channels:** Tmax minus max(local seasonal p90,30°C); log(1+daily rainfall in mm); trailing90 rainfall/(local seasonal p20+1 mm); seasonal sine; seasonal cosine. The climate reference is known before supervised prediction years. No target-time observations, centered weather smoothers or future-window imputations enter model inputs.

**Baselines:** antecedent compound-state persistence, training-only seasonal event climatology, and logistic regression on summaries of exactly the same 56-day feature window. Persistence predicts t+7 by the state at t; it is not allowed to see the label at t+7.

**Episode:** a maximal consecutive run of compound-labelled days in a cell. Test-window endpoints can truncate episodes. Predicted/labelled episodes are matched one-to-one by temporal overlap. True-negative episodes have no natural denominator and are left undefined. Spatially connected episodes across adjacent cells are additionally counted as a dependence diagnostic; neither cell-days nor cell episodes are automatically independent observations.
