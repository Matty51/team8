# VPS Data Warehouse — Architecture Plan

**Purpose:** Centralized data store accessible from 3 personal machines (laptop, work desktop, future home computer) for SNB financials, business CSVs, and a star map dataset. Queryable via Claude/MCP. Networked privately via Tailscale.

---

## 1. Core Decisions

| Decision | Choice | Why |
|---|---|---|
| OS | Ubuntu 24.04 LTS | Long support window, best Docker/MCP ecosystem |
| Networking | Tailscale (private mesh) | No public ports, only your 3 devices |
| Containerization | Docker + docker compose | Easy to rebuild, version everything |
| Primary DB | PostgreSQL 16 | Handles structured business data + JSON |
| Vector DB | Qdrant | For semantic search across docs/CSVs |
| File storage | Direct mount on 30TB volume | Raw CSVs, exports, backups |
| MCP transport | SSE over Tailscale | Encrypted, no public exposure |
| Reverse proxy | Caddy | Auto-TLS even on Tailscale, simple config |
| Backups | restic → external location | Encrypted, deduplicated |

---

## 2. Network Topology

```
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│  Laptop (Mac)   │      │  Work Desktop   │      │  Home Computer  │
│  Tailscale IP   │      │  Tailscale IP   │      │  Tailscale IP   │
└────────┬────────┘      └────────┬────────┘      └────────┬────────┘
         │                        │                        │
         └────────────────────────┼────────────────────────┘
                                  │  (Tailscale mesh, WireGuard)
                                  ▼
                       ┌──────────────────────┐
                       │   VPS (Tailscale)    │
                       │   100.x.y.z          │
                       │                      │
                       │  ┌────────────────┐  │
                       │  │ Caddy (443)    │  │  ← Tailscale-only listener
                       │  └───────┬────────┘  │
                       │          │           │
                       │  ┌───────▼────────┐  │
                       │  │  MCP Server    │  │  ← serves PG + Qdrant + Files
                       │  └───┬────┬───┬───┘  │
                       │      │    │   │      │
                       │  ┌───▼─┐ ┌▼──┐ ┌▼──┐ │
                       │  │ PG  │ │Qd │ │Fs │ │
                       │  └─────┘ └───┘ └───┘ │
                       └──────────────────────┘
```

**Key principle:** the VPS should have its public ssh port firewalled to Tailscale only after initial setup. Nothing should be reachable from the open internet except Tailscale's own coordination.

---

## 3. Directory Layout (on VPS)

```
/opt/warehouse/
├── docker-compose.yml          # Master compose file
├── .env                        # Secrets (NOT in git)
├── caddy/
│   └── Caddyfile
├── postgres/
│   ├── data/                   # PG data directory (bind mount)
│   └── init/                   # SQL run on first boot
├── qdrant/
│   └── storage/
├── mcp/
│   ├── server.py               # Custom MCP server
│   └── requirements.txt
└── backups/
    └── restic-cache/

/data/                          # 30TB volume mount point
├── raw/                        # Untouched source files
│   ├── snb/                    # Stars N Bars CSVs
│   ├── seere/                  # Other business data
│   └── starmap/                # Star catalog data
├── processed/                  # Cleaned, normalized
├── exports/                    # Reports, generated outputs
└── archive/                    # Cold storage
```

**Why split `/opt/warehouse` from `/data`:** the app config lives on the OS disk (small, fast, easy to snapshot). The actual data lives on the big volume. If you ever need to migrate or resize the data volume, the app stack is unaffected.

---

## 4. Service Stack (docker-compose outline)

Services to define:
- **caddy** — reverse proxy, listens on Tailscale IP only
- **postgres** — main relational DB, port 5432 internal only
- **qdrant** — vector DB, port 6333 internal only
- **mcp-server** — your custom MCP endpoint
- **pgadmin** (optional) — web UI for PG, useful for casual SQL exploration
- **metabase** or **grafana** (optional) — for visualization of SNB financials

All services on a single internal Docker network. Only Caddy exposes ports, and only on the Tailscale interface.

---

## 5. MCP Server Responsibilities

Your custom MCP server should expose tools for:
1. **`query_sql`** — run read-only SQL against PostgreSQL
2. **`search_vectors`** — semantic search across Qdrant collections
3. **`list_files`** — browse `/data/`
4. **`read_file`** — read CSV/text from `/data/`
5. **`ingest_csv`** — load a new CSV into PG with auto-schema detection
6. **`run_report`** — execute a saved query template (for SNB monthly reports etc.)

Keep it small. Each tool does one thing.

---

## 6. Security Checklist (Day 1)

- [ ] Disable root SSH login
- [ ] SSH key-only, no password auth
- [ ] UFW firewall: deny all incoming except SSH (22) initially
- [ ] Install Tailscale, authenticate
- [ ] Once Tailscale works, restrict SSH to Tailscale IP only
- [ ] Caddy binds to Tailscale interface explicitly (not 0.0.0.0)
- [ ] PostgreSQL `pg_hba.conf` allows only Docker network
- [ ] All passwords in `.env`, file mode 600
- [ ] Set up unattended-upgrades for security patches

---

## 7. Backup Strategy

**3-2-1 rule:** 3 copies, 2 different media, 1 offsite.

- **Live:** /data on VPS volume
- **Local snapshot:** restic to a separate VPS volume, daily
- **Offsite:** restic to Backblaze B2 or Wasabi, weekly
- **Critical-only fast restore:** PG dumps to `/data/exports/db-dumps/`, daily

Test restore process at least once before trusting it.

---

## 8. Build Sequence (when you arrive in London)

1. Provision VPS, basic OS hardening, SSH key
2. Install Tailscale, verify all 3 machines can reach VPS
3. Lock down firewall to Tailscale-only
4. Mount 30TB volume at `/data`, set up directory structure
5. Install Docker + docker compose
6. Bring up Postgres alone, verify, set passwords
7. Bring up Qdrant alone, verify
8. Build and test MCP server locally first
9. Add Caddy, point Claude clients at the MCP endpoint
10. Load first dataset (suggest: SNB financials — small, you know the shape)
11. Set up backups
12. Document the whole thing in a README in `/opt/warehouse`

---

## 9. Open Questions to Resolve Before Build

- VPS provider? (Hetzner gives best storage/$ for 30TB; OVH and Contabo also options)
- Will the 30TB be a single volume or RAID? (RAID1 doubles cost but saves you on disk failure)
- Tailscale free tier covers up to 100 devices — fine for now
- Do you want web dashboards (Metabase) or only Claude/MCP access?
