"""Scripts package for GrowNXT Server data ingestion and utility CLI jobs.

The re-exports below are optional. Importing them eagerly makes the whole
package -- and therefore every ``python -m scripts.<name>`` invocation -- fail
when one legacy module is absent, which is why they are guarded.
"""

__all__ = []

try:
    from scripts.fetch_financial_data import process_stock_ticker
    __all__.append("process_stock_ticker")
except ImportError:  # pragma: no cover - optional legacy job
    pass

try:
    from scripts.fetch_nifty50 import fetch_all_nifty50
    __all__.append("fetch_all_nifty50")
except ImportError:  # pragma: no cover - optional legacy job
    pass
