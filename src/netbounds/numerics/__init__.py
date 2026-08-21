"""Private fixed 1D numerical kernels; the public API is the CLI only."""

from .cases import CASES, case_names
from .compare import compare_reproduction

__all__ = ["CASES", "case_names", "compare_reproduction"]
