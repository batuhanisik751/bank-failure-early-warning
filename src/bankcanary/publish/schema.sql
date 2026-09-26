-- BankCanary published schema (CONTRACT section 16). Python owns this DDL; the web app
-- mirrors it read-only. Every statement is idempotent so the file can be applied on every
-- publish. Money is in thousands of dollars as in the warehouse; numeric columns are
-- double precision. Educational project, not a credit rating, not investment advice, not a
-- supervisory assessment; FDIC insurance covers $250,000 per depositor, per bank, per
-- ownership category. The ratio and peer-percentile columns of ``ratios`` are ``real``
-- (CONTRACT 16 deviation, recorded there) to keep the database under the 400 MB budget.

CREATE TABLE IF NOT EXISTS model_versions (
    model_version     text PRIMARY KEY,
    model             text NOT NULL,
    train_end_repdte  date,
    git_sha           text,
    trained_at        timestamptz,
    features_version  text,
    notes             text
);

CREATE TABLE IF NOT EXISTS banks (
    cert                 bigint PRIMARY KEY,
    name                 text,
    city                 text,
    state                text,
    bkclass              text,
    charter_class_label  text,
    estymd               date,
    endefymd             date,
    active               boolean,
    fed_rssd             bigint,
    rssdhcr              bigint,
    holding_company_name text,
    latitude             double precision,
    longitude            double precision,
    latest_assets        double precision,
    size_bucket          text,
    exit_reason          text,
    fail_date            date
);
CREATE INDEX IF NOT EXISTS banks_state_idx ON banks (state);
CREATE INDEX IF NOT EXISTS banks_fed_rssd_idx ON banks (fed_rssd);
CREATE INDEX IF NOT EXISTS banks_rssdhcr_idx ON banks (rssdhcr);

CREATE TABLE IF NOT EXISTS quarters (
    repdte              date PRIMARY KEY,
    label               text NOT NULL,
    avail_date          date,
    n_banks             integer,
    n_failures_next_4q  integer,
    label_complete_4q   boolean,
    model_year          integer,
    model_version       text
);

CREATE TABLE IF NOT EXISTS scores (
    cert                bigint NOT NULL,
    repdte              date NOT NULL,
    horizon             integer NOT NULL,
    rank                integer,
    score               double precision,
    probability         real,
    percentile          real,
    delta_prob_prior_q  real,
    model               text NOT NULL,
    band                text,
    model_version       text,
    PRIMARY KEY (cert, repdte, model)
);
CREATE INDEX IF NOT EXISTS scores_repdte_rank_idx ON scores (repdte, rank);

CREATE TABLE IF NOT EXISTS drivers (
    cert           bigint NOT NULL,
    repdte         date NOT NULL,
    model          text NOT NULL,
    rank           integer NOT NULL,
    feature        text,
    feature_label  text,
    shap_value     real,
    feature_value  real,
    direction      text,
    PRIMARY KEY (cert, repdte, model, rank)
);
CREATE INDEX IF NOT EXISTS drivers_repdte_idx ON drivers (repdte);

CREATE TABLE IF NOT EXISTS ratios (
    cert                          bigint NOT NULL,
    repdte                        date NOT NULL,
    total_assets                  double precision,
    equity_to_assets              real,
    equity_to_assets_pct          real,
    tier1_leverage                real,
    tier1_leverage_pct            real,
    noncurrent_ratio              real,
    noncurrent_ratio_pct          real,
    npa_to_assets                 real,
    npa_to_assets_pct             real,
    texas_ratio                   real,
    texas_ratio_pct               real,
    roa_q                         real,
    roa_q_pct                     real,
    nim_q                         real,
    nim_q_pct                     real,
    efficiency_ratio              real,
    efficiency_ratio_pct          real,
    brokered_share                real,
    brokered_share_pct            real,
    uninsured_share               real,
    uninsured_share_pct           real,
    unrealized_loss_to_tier1      real,
    unrealized_loss_to_tier1_pct  real,
    construction_to_capital       real,
    construction_to_capital_pct   real,
    PRIMARY KEY (cert, repdte)
);
CREATE INDEX IF NOT EXISTS ratios_repdte_idx ON ratios (repdte);

CREATE TABLE IF NOT EXISTS peer_stats (
    repdte       date NOT NULL,
    size_bucket  text NOT NULL,
    region       text NOT NULL,
    ratio        text NOT NULL,
    p10          double precision,
    p50          double precision,
    p90          double precision,
    n            integer,
    PRIMARY KEY (repdte, size_bucket, region, ratio)
);

CREATE TABLE IF NOT EXISTS failures (
    cert       bigint NOT NULL,
    fail_date  date NOT NULL,
    name       text,
    city       text,
    state      text,
    restype1   text,
    cost       double precision,
    qbfasset   double precision,
    qbfdep     double precision,
    PRIMARY KEY (cert, fail_date)
);
CREATE INDEX IF NOT EXISTS failures_fail_date_idx ON failures (fail_date);

CREATE TABLE IF NOT EXISTS walkforward_metrics (
    model             text NOT NULL,
    horizon           integer NOT NULL,
    test_year         integer NOT NULL,
    n                 integer,
    n_failures        integer,
    pr_auc            double precision,
    pr_auc_lo         double precision,
    pr_auc_hi         double precision,
    recall_at_2pct    double precision,
    recall_lo         double precision,
    recall_hi         double precision,
    roc_auc           double precision,
    brier_raw         double precision,
    brier_calibrated  double precision,
    low_confidence    boolean,
    PRIMARY KEY (model, horizon, test_year)
);

CREATE TABLE IF NOT EXISTS case_study_2023 (
    cert         bigint NOT NULL,
    quarter      text NOT NULL,
    view         text NOT NULL,
    model        text NOT NULL,
    probability  double precision,
    rank         integer,
    percentile   double precision,
    n_scored     integer,
    PRIMARY KEY (cert, quarter, view, model)
);

CREATE TABLE IF NOT EXISTS case_study_series (
    cert                      bigint NOT NULL,
    repdte                    date NOT NULL,
    unrealized_loss_to_tier1  double precision,
    uninsured_share           double precision,
    peer_p50_unrealized       double precision,
    peer_p05_unrealized       double precision,
    peer_p50_uninsured        double precision,
    peer_p95_uninsured        double precision,
    PRIMARY KEY (cert, repdte)
);

CREATE TABLE IF NOT EXISTS rate_shock_scores (
    cert                      bigint NOT NULL,
    shock_bp                  integer NOT NULL,
    duration_years            integer NOT NULL,
    extra_loss                double precision,
    adjusted_tier1_leverage   double precision,
    unrealized_loss_to_tier1  double precision,
    probability               double precision,
    rank                      integer,
    band                      text,
    PRIMARY KEY (cert, shock_bp, duration_years)
);

CREATE TABLE IF NOT EXISTS map_quarters (
    repdte               date NOT NULL,
    cert                 bigint NOT NULL,
    latitude             real,
    longitude            real,
    band                 text,
    probability          real,
    failed_this_quarter  boolean,
    PRIMARY KEY (repdte, cert)
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id         text PRIMARY KEY,
    started_at     timestamptz,
    finished_at    timestamptz,
    status         text,
    latest_repdte  date,
    new_quarter    boolean,
    rows_written   jsonb,
    log            text
);

-- The disclaimer travels with every table.
COMMENT ON TABLE model_versions IS 'Educational project, not a credit rating, not investment advice, not a supervisory assessment. FDIC insurance covers $250,000 per depositor, per bank, per ownership category.';
COMMENT ON TABLE banks IS 'Educational project, not a credit rating, not investment advice, not a supervisory assessment. FDIC insurance covers $250,000 per depositor, per bank, per ownership category.';
COMMENT ON TABLE quarters IS 'Educational project, not a credit rating, not investment advice, not a supervisory assessment. FDIC insurance covers $250,000 per depositor, per bank, per ownership category.';
COMMENT ON TABLE scores IS 'Educational project, not a credit rating, not investment advice, not a supervisory assessment. FDIC insurance covers $250,000 per depositor, per bank, per ownership category.';
COMMENT ON TABLE drivers IS 'Educational project, not a credit rating, not investment advice, not a supervisory assessment. FDIC insurance covers $250,000 per depositor, per bank, per ownership category.';
COMMENT ON TABLE ratios IS 'Educational project, not a credit rating, not investment advice, not a supervisory assessment. FDIC insurance covers $250,000 per depositor, per bank, per ownership category.';
COMMENT ON TABLE peer_stats IS 'Educational project, not a credit rating, not investment advice, not a supervisory assessment. FDIC insurance covers $250,000 per depositor, per bank, per ownership category.';
COMMENT ON TABLE failures IS 'Educational project, not a credit rating, not investment advice, not a supervisory assessment. FDIC insurance covers $250,000 per depositor, per bank, per ownership category.';
COMMENT ON TABLE walkforward_metrics IS 'Educational project, not a credit rating, not investment advice, not a supervisory assessment. FDIC insurance covers $250,000 per depositor, per bank, per ownership category.';
COMMENT ON TABLE case_study_2023 IS 'Educational project, not a credit rating, not investment advice, not a supervisory assessment. FDIC insurance covers $250,000 per depositor, per bank, per ownership category.';
COMMENT ON TABLE case_study_series IS 'Educational project, not a credit rating, not investment advice, not a supervisory assessment. FDIC insurance covers $250,000 per depositor, per bank, per ownership category.';
COMMENT ON TABLE rate_shock_scores IS 'Educational project, not a credit rating, not investment advice, not a supervisory assessment. FDIC insurance covers $250,000 per depositor, per bank, per ownership category.';
COMMENT ON TABLE map_quarters IS 'Educational project, not a credit rating, not investment advice, not a supervisory assessment. FDIC insurance covers $250,000 per depositor, per bank, per ownership category.';
COMMENT ON TABLE pipeline_runs IS 'Educational project, not a credit rating, not investment advice, not a supervisory assessment. FDIC insurance covers $250,000 per depositor, per bank, per ownership category.';
