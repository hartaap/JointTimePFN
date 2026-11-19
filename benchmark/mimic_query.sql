-- BigQuery export behind mimic_iv_data/bq-results-*.csv (MIMIC-IV v3.1, PhysioNet credentialed access)
-- hourly means of chartevents items on a complete hourly grid, ICU stays of at least 192 hours long
-- our benchmark/mimic_prepare.py keeps 10 of these columns, range-filters them and bins them to 3 h afterwards
WITH icu_hours AS (
  SELECT
    stay_id,
    intime,
    hour_grid
  FROM `physionet-data.mimiciv_3_1_icu.icustays`,
  UNNEST(GENERATE_TIMESTAMP_ARRAY(
    TIMESTAMP_TRUNC(CAST(intime AS TIMESTAMP), HOUR),
    TIMESTAMP_TRUNC(CAST(outtime AS TIMESTAMP), HOUR),
    INTERVAL 1 HOUR
  )) AS hour_grid
  WHERE DATETIME_DIFF(outtime, intime, HOUR) >= 192
),
hourly_vitals AS (
  SELECT
    ce.stay_id,
    TIMESTAMP_TRUNC(CAST(ce.charttime AS TIMESTAMP), HOUR) AS chart_hour,
    AVG(CASE WHEN itemid IN (220045, 223764, 223765, 224647) AND valuenum > 0 AND valuenum < 300 THEN valuenum END) AS heart_rate,
    AVG(CASE WHEN itemid IN (225309, 220050, 220179, 224167, 227243, 224645, 224646, 223766) AND valuenum > 0 AND valuenum < 400 THEN valuenum END) AS sbp,
    AVG(CASE WHEN itemid IN (225310, 220051, 220180, 224643, 227242, 226092, 226094, 226096) AND valuenum > 0 AND valuenum < 300 THEN valuenum END) AS dbp,
    AVG(CASE WHEN itemid IN (225312, 220052, 220181) AND valuenum > 0 AND valuenum < 300 THEN valuenum END) AS mbp,
    AVG(CASE WHEN itemid = 224166 AND valuenum > 0 THEN valuenum END) AS doppler_bp,
    AVG(CASE WHEN itemid IN (220210, 224690) AND valuenum > 0 AND valuenum < 70 THEN valuenum END) AS resp_rate,
    AVG(CASE WHEN itemid = 220277 AND valuenum > 0 AND valuenum <= 100 THEN valuenum END) AS spo2,
    ROUND(AVG(
        CASE
            WHEN itemid = 223761 AND valuenum > 70 AND valuenum < 120 THEN (valuenum - 32) / 1.8
            WHEN itemid IN (223762, 227632, 227634, 226329) AND valuenum > 10 AND valuenum < 50 THEN valuenum
        END), 2) AS temperature,
    AVG(CASE WHEN itemid = 228640 AND valuenum > 0 THEN valuenum END) AS etco2,
    AVG(CASE WHEN itemid = 223763 AND valuenum > 0 THEN valuenum END) AS bladder_pressure,
    AVG(CASE WHEN itemid = 224359 AND valuenum > 0 THEN valuenum END) AS qtc,
    AVG(CASE WHEN itemid = 224192 AND valuenum > 0 THEN valuenum END) AS pulsus_paradoxus,
    AVG(CASE WHEN itemid IN (225664, 220621, 226537) AND valuenum > 0 THEN valuenum END) AS glucose,
    AVG(CASE WHEN itemid IN (220074, 227689) AND valuenum > 0 THEN valuenum END) AS cvp,
    AVG(CASE WHEN itemid IN (224696, 228369) AND valuenum > 0 THEN valuenum END) AS cardiac_output,
    AVG(CASE WHEN itemid = 223835 AND valuenum > 0 THEN valuenum END) AS fio2,
    AVG(CASE WHEN itemid IN (220339, 224700) AND valuenum > 0 THEN valuenum END) AS peep
  FROM `physionet-data.mimiciv_3_1_icu.chartevents` ce
  WHERE ce.itemid IN (
    220045, 223764, 223765, 224647, 225309, 220050, 220179, 224167, 227243, 224645, 224646, 223766,
    225310, 220051, 220180, 224643, 227242, 226092, 226094, 226096, 225312, 220052, 220181,
    224166, 220210, 224690, 220277, 223761, 223762, 227632, 227634, 226329, 228640, 223763,
    224359, 224192, 225664, 220621, 226537, 220074, 227689, 224696, 228369, 223835, 220339, 224700
  )
  GROUP BY ce.stay_id, chart_hour
)
SELECT
  h.stay_id,
  h.hour_grid AS chart_hour,
  TIMESTAMP_DIFF(h.hour_grid, TIMESTAMP_TRUNC(CAST(h.intime AS TIMESTAMP), HOUR), HOUR) AS hours_since_admission,
  v.heart_rate,
  v.sbp,
  v.dbp,
  v.mbp,
  v.doppler_bp,
  v.resp_rate,
  v.spo2,
  v.temperature,
  v.etco2,
  v.bladder_pressure,
  v.qtc,
  v.pulsus_paradoxus,
  v.glucose,
  v.cvp,
  v.cardiac_output,
  v.fio2,
  v.peep
FROM icu_hours h
LEFT JOIN hourly_vitals v
  ON h.stay_id = v.stay_id
  AND h.hour_grid = v.chart_hour
ORDER BY h.stay_id, h.hour_grid;
