/**
 * Read-only Drizzle mirror of src/bankcanary/publish/schema.sql (CONTRACT section 16).
 * Python owns the DDL; this file never migrates anything. tests/schema.test.ts parses the
 * SQL file and fails when a table, column or column type here drifts from it.
 *
 * Dates are read as ISO strings (YYYY-MM-DD) and bigints as numbers (certificate numbers
 * and RSSD ids fit comfortably in 2^53). Money is in thousands of dollars.
 *
 * Educational project, not a credit rating, not investment advice, not a supervisory
 * assessment. FDIC insurance covers $250,000 per depositor, per bank, per ownership category.
 */
import {
  bigint,
  boolean,
  date,
  doublePrecision,
  integer,
  jsonb,
  pgTable,
  primaryKey,
  real,
  text,
  timestamp,
} from "drizzle-orm/pg-core";

const cert = () => bigint("cert", { mode: "number" });
const repdte = () => date("repdte", { mode: "string" });

export const modelVersions = pgTable("model_versions", {
  modelVersion: text("model_version").primaryKey(),
  model: text("model").notNull(),
  trainEndRepdte: date("train_end_repdte", { mode: "string" }),
  gitSha: text("git_sha"),
  trainedAt: timestamp("trained_at", { withTimezone: true, mode: "string" }),
  featuresVersion: text("features_version"),
  notes: text("notes"),
});

export const banks = pgTable("banks", {
  cert: cert().primaryKey(),
  name: text("name"),
  city: text("city"),
  state: text("state"),
  bkclass: text("bkclass"),
  charterClassLabel: text("charter_class_label"),
  estymd: date("estymd", { mode: "string" }),
  endefymd: date("endefymd", { mode: "string" }),
  active: boolean("active"),
  fedRssd: bigint("fed_rssd", { mode: "number" }),
  rssdhcr: bigint("rssdhcr", { mode: "number" }),
  holdingCompanyName: text("holding_company_name"),
  latitude: doublePrecision("latitude"),
  longitude: doublePrecision("longitude"),
  latestAssets: doublePrecision("latest_assets"),
  sizeBucket: text("size_bucket"),
  exitReason: text("exit_reason"),
  failDate: date("fail_date", { mode: "string" }),
});

export const quarters = pgTable("quarters", {
  repdte: repdte().primaryKey(),
  label: text("label").notNull(),
  availDate: date("avail_date", { mode: "string" }),
  nBanks: integer("n_banks"),
  nFailuresNext4q: integer("n_failures_next_4q"),
  labelComplete4q: boolean("label_complete_4q"),
  modelYear: integer("model_year"),
  modelVersion: text("model_version"),
});

export const scores = pgTable(
  "scores",
  {
    cert: cert().notNull(),
    repdte: repdte().notNull(),
    horizon: integer("horizon").notNull(),
    rank: integer("rank"),
    score: doublePrecision("score"),
    probability: real("probability"),
    percentile: real("percentile"),
    deltaProbPriorQ: real("delta_prob_prior_q"),
    model: text("model").notNull(),
    band: text("band"),
    modelVersion: text("model_version"),
  },
  (t) => [primaryKey({ columns: [t.cert, t.repdte, t.model] })],
);

export const drivers = pgTable(
  "drivers",
  {
    cert: cert().notNull(),
    repdte: repdte().notNull(),
    model: text("model").notNull(),
    rank: integer("rank").notNull(),
    feature: text("feature"),
    featureLabel: text("feature_label"),
    shapValue: real("shap_value"),
    featureValue: real("feature_value"),
    direction: text("direction"),
  },
  (t) => [primaryKey({ columns: [t.cert, t.repdte, t.model, t.rank] })],
);
export const ratios = pgTable(
  "ratios",
  {
    cert: cert().notNull(),
    repdte: repdte().notNull(),
    totalAssets: doublePrecision("total_assets"),
    equityToAssets: real("equity_to_assets"),
    equityToAssetsPct: real("equity_to_assets_pct"),
    tier1Leverage: real("tier1_leverage"),
    tier1LeveragePct: real("tier1_leverage_pct"),
    noncurrentRatio: real("noncurrent_ratio"),
    noncurrentRatioPct: real("noncurrent_ratio_pct"),
    npaToAssets: real("npa_to_assets"),
    npaToAssetsPct: real("npa_to_assets_pct"),
    texasRatio: real("texas_ratio"),
    texasRatioPct: real("texas_ratio_pct"),
    roaQ: real("roa_q"),
    roaQPct: real("roa_q_pct"),
    nimQ: real("nim_q"),
    nimQPct: real("nim_q_pct"),
    efficiencyRatio: real("efficiency_ratio"),
    efficiencyRatioPct: real("efficiency_ratio_pct"),
    brokeredShare: real("brokered_share"),
    brokeredSharePct: real("brokered_share_pct"),
    uninsuredShare: real("uninsured_share"),
    uninsuredSharePct: real("uninsured_share_pct"),
    unrealizedLossToTier1: real("unrealized_loss_to_tier1"),
    unrealizedLossToTier1Pct: real("unrealized_loss_to_tier1_pct"),
    constructionToCapital: real("construction_to_capital"),
    constructionToCapitalPct: real("construction_to_capital_pct"),
  },
  (t) => [primaryKey({ columns: [t.cert, t.repdte] })],
);

export const peerStats = pgTable(
  "peer_stats",
  {
    repdte: repdte().notNull(),
    sizeBucket: text("size_bucket").notNull(),
    region: text("region").notNull(),
    ratio: text("ratio").notNull(),
    p10: doublePrecision("p10"),
    p50: doublePrecision("p50"),
    p90: doublePrecision("p90"),
    n: integer("n"),
  },
  (t) => [primaryKey({ columns: [t.repdte, t.sizeBucket, t.region, t.ratio] })],
);

export const failures = pgTable(
  "failures",
  {
    cert: cert().notNull(),
    failDate: date("fail_date", { mode: "string" }).notNull(),
    name: text("name"),
    city: text("city"),
    state: text("state"),
    restype1: text("restype1"),
    cost: doublePrecision("cost"),
    qbfasset: doublePrecision("qbfasset"),
    qbfdep: doublePrecision("qbfdep"),
  },
  (t) => [primaryKey({ columns: [t.cert, t.failDate] })],
);

export const walkforwardMetrics = pgTable(
  "walkforward_metrics",
  {
    model: text("model").notNull(),
    horizon: integer("horizon").notNull(),
    testYear: integer("test_year").notNull(),
    n: integer("n"),
    nFailures: integer("n_failures"),
    prAuc: doublePrecision("pr_auc"),
    prAucLo: doublePrecision("pr_auc_lo"),
    prAucHi: doublePrecision("pr_auc_hi"),
    recallAt2pct: doublePrecision("recall_at_2pct"),
    recallLo: doublePrecision("recall_lo"),
    recallHi: doublePrecision("recall_hi"),
    rocAuc: doublePrecision("roc_auc"),
    brierRaw: doublePrecision("brier_raw"),
    brierCalibrated: doublePrecision("brier_calibrated"),
    lowConfidence: boolean("low_confidence"),
  },
  (t) => [primaryKey({ columns: [t.model, t.horizon, t.testYear] })],
);
export const caseStudy2023 = pgTable(
  "case_study_2023",
  {
    cert: cert().notNull(),
    quarter: text("quarter").notNull(),
    view: text("view").notNull(),
    model: text("model").notNull(),
    probability: doublePrecision("probability"),
    rank: integer("rank"),
    percentile: doublePrecision("percentile"),
    nScored: integer("n_scored"),
  },
  (t) => [primaryKey({ columns: [t.cert, t.quarter, t.view, t.model] })],
);

export const caseStudySeries = pgTable(
  "case_study_series",
  {
    cert: cert().notNull(),
    repdte: repdte().notNull(),
    unrealizedLossToTier1: doublePrecision("unrealized_loss_to_tier1"),
    uninsuredShare: doublePrecision("uninsured_share"),
    peerP50Unrealized: doublePrecision("peer_p50_unrealized"),
    peerP05Unrealized: doublePrecision("peer_p05_unrealized"),
    peerP50Uninsured: doublePrecision("peer_p50_uninsured"),
    peerP95Uninsured: doublePrecision("peer_p95_uninsured"),
  },
  (t) => [primaryKey({ columns: [t.cert, t.repdte] })],
);

export const rateShockScores = pgTable(
  "rate_shock_scores",
  {
    cert: cert().notNull(),
    shockBp: integer("shock_bp").notNull(),
    durationYears: integer("duration_years").notNull(),
    extraLoss: doublePrecision("extra_loss"),
    adjustedTier1Leverage: doublePrecision("adjusted_tier1_leverage"),
    unrealizedLossToTier1: doublePrecision("unrealized_loss_to_tier1"),
    probability: doublePrecision("probability"),
    rank: integer("rank"),
    band: text("band"),
  },
  (t) => [primaryKey({ columns: [t.cert, t.shockBp, t.durationYears] })],
);

export const mapQuarters = pgTable(
  "map_quarters",
  {
    repdte: repdte().notNull(),
    cert: cert().notNull(),
    latitude: real("latitude"),
    longitude: real("longitude"),
    band: text("band"),
    probability: real("probability"),
    failedThisQuarter: boolean("failed_this_quarter"),
  },
  (t) => [primaryKey({ columns: [t.repdte, t.cert] })],
);

export const pipelineRuns = pgTable("pipeline_runs", {
  runId: text("run_id").primaryKey(),
  startedAt: timestamp("started_at", { withTimezone: true, mode: "string" }),
  finishedAt: timestamp("finished_at", { withTimezone: true, mode: "string" }),
  status: text("status"),
  latestRepdte: date("latest_repdte", { mode: "string" }),
  newQuarter: boolean("new_quarter"),
  rowsWritten: jsonb("rows_written"),
  log: text("log"),
});

/** Every published table, keyed by its SQL name, for the drift test and the client. */
export const tables = {
  model_versions: modelVersions,
  banks,
  quarters,
  scores,
  drivers,
  ratios,
  peer_stats: peerStats,
  failures,
  walkforward_metrics: walkforwardMetrics,
  case_study_2023: caseStudy2023,
  case_study_series: caseStudySeries,
  rate_shock_scores: rateShockScores,
  map_quarters: mapQuarters,
  pipeline_runs: pipelineRuns,
} as const;
