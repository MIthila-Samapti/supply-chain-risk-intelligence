# Supply Chain Disruption Risk Intelligence

A working proof-of-concept for the risk-intelligence piece of my broader
AI-Powered Predictive Supply Chain Resilience Framework — this one flags
demand disruptions, turns them into a composite risk score per
store/product category, and feeds that score back into inventory planning
so high-risk categories carry more buffer.

It builds directly on my earlier
[retail-demand-forecasting](https://github.com/MIthila-Samapti/retail-demand-forecasting)
project, reusing its cleaned daily sales panel rather than starting from
raw data again.

## Problem

A forecast tells you what you expect to sell. It doesn't tell you how much
to trust that number for a given product this week, or which categories
are quietly becoming higher-risk to stock. This project adds that layer:
detect unexplained demand shocks, score disruption risk per
store/category, and translate that score into a concrete inventory
decision.

## Data

`data/processed/daily_sales_by_store_family.csv` — the same panel from the
demand-forecasting project (10 stores, 8 product families, 2013-2017),
already carrying promotion, holiday, and oil-price context. See
`data/README.md`.

## Methodology (in progress)

1. **Disruption detection** — flag demand shocks in the daily sales series
   that aren't explained by known promotions or holidays, using rolling
   z-scores and change-point detection (`ruptures`) as a cross-check.
2. **Composite risk scoring** — combine disruption frequency/severity with
   an external volatility proxy (oil price volatility) into a single risk
   score per store/category.
3. **Risk-adjusted inventory** — extend the safety-stock formula from the
   demand-forecasting project with a risk multiplier, so higher-risk
   categories carry more buffer stock, and quantify the cost/service-level
   trade-off versus a risk-blind policy.
4. **Reporting** — tidy exports for Tableau/Power BI and a summary
   dashboard, same pattern as the first project.

## Project structure

```
src/            core Python modules
notebooks/      exploration and write-ups, in order
reports/        result exports and figures
tests/          unit tests (pytest)
data/           processed data (raw data, if any is added, is not committed)
```

## Setup

```bash
python -m venv venv
source venv/bin/activate  # venv\Scripts\activate on Windows
pip install -r requirements.txt
pytest
```

## Author

Mithila Zaman Samapti — [retail-demand-forecasting](https://github.com/MIthila-Samapti/retail-demand-forecasting) is the companion project this one builds on.
