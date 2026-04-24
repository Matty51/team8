# VPS Data Warehouse — Schema Plan

## Design Philosophy

Two layers:
1. **Raw layer** — exact mirror of source CSVs, one schema per source. Never modified after load. This is your "source of truth."
2. **Curated layer** — cleaned, joined, normalized tables that you actually query. Derived from raw layer via SQL views or materialized views.

If you ever mess up curated, raw is still there. If you ever change how you want to clean data, you don't re-pull from sources — you just rebuild curated.

---

## PostgreSQL Schema Layout

```sql
-- Top-level schemas (namespaces)
CREATE SCHEMA raw_snb;        -- Stars N Bars raw imports
CREATE SCHEMA raw_seere;      -- Other Seere Group entities
CREATE SCHEMA raw_starmap;    -- Star catalog data
CREATE SCHEMA curated;        -- Cleaned, query-ready tables
CREATE SCHEMA reports;        -- Materialized views for dashboards
CREATE SCHEMA meta;           -- Metadata about ingested files
```

---

## Meta Tables (track what's been loaded)

```sql
-- Every CSV/file ingested gets logged here
CREATE TABLE meta.ingestion_log (
    id              BIGSERIAL PRIMARY KEY,
    source_path     TEXT NOT NULL,           -- /data/raw/snb/2026-03-sales.csv
    target_schema   TEXT NOT NULL,           -- raw_snb
    target_table    TEXT NOT NULL,           -- sales_2026_03
    row_count       BIGINT,
    file_hash       TEXT,                    -- sha256, detect re-loads
    file_bytes      BIGINT,
    ingested_at     TIMESTAMPTZ DEFAULT NOW(),
    ingested_by     TEXT DEFAULT CURRENT_USER,
    notes           TEXT
);

CREATE INDEX ix_ingestion_log_hash ON meta.ingestion_log(file_hash);
CREATE INDEX ix_ingestion_log_target ON meta.ingestion_log(target_schema, target_table);
```

---

## SNB Financial Tables (curated layer, sketch)

These are the ones you'll actually query. Adjust as you see your real CSVs.

```sql
-- Daily sales rollup
CREATE TABLE curated.snb_daily_sales (
    business_date   DATE NOT NULL,
    revenue_aed     NUMERIC(12,2) NOT NULL,
    covers          INTEGER,                 -- # of guests
    avg_check_aed   NUMERIC(8,2) GENERATED ALWAYS AS
                       (CASE WHEN covers > 0 THEN revenue_aed / covers END) STORED,
    food_revenue    NUMERIC(12,2),
    beverage_revenue NUMERIC(12,2),
    notes           TEXT,
    PRIMARY KEY (business_date)
);

-- Category-level breakdown
CREATE TABLE curated.snb_sales_by_category (
    business_date   DATE NOT NULL,
    category        TEXT NOT NULL,           -- 'Food', 'Beer', 'Wine', 'Spirits', etc.
    subcategory     TEXT,
    revenue_aed     NUMERIC(12,2) NOT NULL,
    units_sold      INTEGER,
    PRIMARY KEY (business_date, category, subcategory)
);

-- Cost / P&L lines
CREATE TABLE curated.snb_costs (
    period_month    DATE NOT NULL,           -- first day of month
    cost_category   TEXT NOT NULL,           -- 'COGS', 'Labor', 'Rent', 'Utilities', etc.
    amount_aed      NUMERIC(12,2) NOT NULL,
    notes           TEXT,
    PRIMARY KEY (period_month, cost_category)
);

CREATE INDEX ix_snb_daily_sales_month
    ON curated.snb_daily_sales (date_trunc('month', business_date));
```

---

## Generic CSV Landing Pattern

For ad-hoc CSVs you upload from any business, a single landing table:

```sql
CREATE TABLE meta.csv_uploads (
    upload_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    business        TEXT NOT NULL,           -- 'snb', 'agefix', 'edt', 'kezad'
    description     TEXT,
    uploaded_at     TIMESTAMPTZ DEFAULT NOW(),
    columns         JSONB,                   -- inferred column names + types
    row_count       BIGINT
);

-- Then each upload creates a dynamic table named e.g.
-- raw_snb.upload_<short_uuid>
-- The MCP ingest_csv tool handles this.
```

---

## Star Map Tables

Adjust based on what catalogs you're actually loading. This is a flexible starting point.

```sql
CREATE TABLE raw_starmap.stars (
    star_id         BIGINT PRIMARY KEY,      -- catalog ID
    catalog         TEXT NOT NULL,           -- 'gaia_dr3', 'hipparcos', etc.
    ra_deg          DOUBLE PRECISION,        -- right ascension
    dec_deg         DOUBLE PRECISION,        -- declination
    parallax_mas    DOUBLE PRECISION,
    distance_pc     DOUBLE PRECISION,
    apparent_mag    REAL,
    abs_mag         REAL,
    spectral_class  TEXT,
    color_bv        REAL,
    proper_name     TEXT,                    -- Sirius, Vega, etc. (mostly NULL)
    notes           TEXT
);

-- Spatial index (if you load PostGIS) or use BRIN for ra/dec
CREATE INDEX ix_stars_radec ON raw_starmap.stars USING BRIN (ra_deg, dec_deg);
CREATE INDEX ix_stars_mag ON raw_starmap.stars (apparent_mag);

-- Star clusters / groupings
CREATE TABLE raw_starmap.clusters (
    cluster_id      SERIAL PRIMARY KEY,
    name            TEXT NOT NULL,           -- 'Pleiades', 'Hyades', etc.
    cluster_type    TEXT,                    -- 'open', 'globular', 'asterism'
    ra_center_deg   DOUBLE PRECISION,
    dec_center_deg  DOUBLE PRECISION,
    distance_pc     DOUBLE PRECISION,
    age_myr         REAL
);

CREATE TABLE raw_starmap.cluster_members (
    cluster_id      INT REFERENCES raw_starmap.clusters(cluster_id),
    star_id         BIGINT REFERENCES raw_starmap.stars(star_id),
    PRIMARY KEY (cluster_id, star_id)
);
```

---

## Qdrant Collection Plan

Collections are independent of PostgreSQL. Use them where semantic search beats SQL.

| Collection | What's in it | Vector source | Use case |
|---|---|---|---|
| `documents` | Business docs, contracts, memos | Embed full text | "Find docs about Star Deck commission" |
| `csv_descriptions` | One vector per uploaded CSV (its description + column list) | Embed metadata | "Which CSV had Q3 marketing spend?" |
| `notes` | Personal notes, meeting notes | Embed each note | "What did Marwan say about the lease?" |
| `stars_named` | Only stars with proper names + descriptions | Embed name+description | "Find stars associated with Egyptian mythology" |

Each Qdrant point payload should include:
```json
{
  "source_path": "/data/raw/...",
  "source_type": "csv|pdf|note|star",
  "ingested_at": "...",
  "business": "snb|seere|...",
  "title": "...",
  "snippet": "first 200 chars"
}
```

---

## Reports / Materialized Views

Pre-compute the things you'll look at often. Refresh nightly.

```sql
CREATE MATERIALIZED VIEW reports.snb_monthly_pnl AS
SELECT
    date_trunc('month', s.business_date)::date AS period_month,
    SUM(s.revenue_aed)                          AS revenue,
    SUM(c.amount_aed) FILTER (WHERE c.cost_category = 'COGS')  AS cogs,
    SUM(c.amount_aed) FILTER (WHERE c.cost_category = 'Labor') AS labor,
    SUM(c.amount_aed)                           AS total_costs,
    SUM(s.revenue_aed) - SUM(c.amount_aed)      AS net
FROM curated.snb_daily_sales s
LEFT JOIN curated.snb_costs c
       ON date_trunc('month', s.business_date) = c.period_month
GROUP BY 1;

CREATE UNIQUE INDEX ON reports.snb_monthly_pnl (period_month);

-- Refresh nightly via cron:
-- REFRESH MATERIALIZED VIEW CONCURRENTLY reports.snb_monthly_pnl;
```

---

## Naming Conventions (lock these in now, save pain later)

- **Schemas:** lowercase, snake_case, prefix with `raw_`, `curated_`, `reports_`, `meta`
- **Tables:** lowercase, snake_case, plural for collections (`stars`), singular for single-row config
- **Money columns:** always include currency in name (`revenue_aed`, `cost_usd`)
- **Dates:** `_date` for DATE, `_at` for TIMESTAMPTZ, `_month` for first-of-month dates
- **Booleans:** prefix with `is_` or `has_`
- **Foreign keys:** `<table_singular>_id`

---

## What to Build First (priority order)

1. `meta.ingestion_log` — you need this from day one to track what you've loaded
2. `meta.csv_uploads` + the generic CSV landing pattern — unblocks loading anything
3. SNB curated tables — your first real workload
4. The MCP `query_sql` and `ingest_csv` tools
5. Qdrant `documents` collection
6. Star map tables (whenever you actually have the data ready)

Don't build curated tables until you've seen the actual raw CSVs. Schema design without seeing real data = wasted work.
