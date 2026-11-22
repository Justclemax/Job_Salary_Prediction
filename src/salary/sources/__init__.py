"""Connectors to real-world salary data.

Each connector downloads a public source once, caches it under `data/external/`
and maps it onto a harmonised schema (see `world.py`):

- `stackoverflow`: Stack Overflow Developer Survey, individual responses (ODbL 1.0).
- `worldbank`: World Bank API, PPP conversion factors and exchange rates.
"""
