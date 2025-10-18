#!/usr/bin/env python3
"""
Base model classes and interfaces
"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Optional
import pandas as pd


class CalculationStatus(Enum):
    """Status of a calculation step."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CalculationResult:
    """Container for calculation results."""

    def __init__(self,
                 status: CalculationStatus,
                 data: Optional[pd.DataFrame] = None,
                 metadata: Optional[Dict[str, Any]] = None,
                 error: Optional[str] = None):
        self.status = status
        self.data = data
        self.metadata = metadata or {}
        self.error = error

    def is_successful(self) -> bool:
        """Check if calculation was successful."""
        return self.status == CalculationStatus.COMPLETED and self.data is not None

    def get_summary(self) -> str:
        """Get a summary of the results."""
        if self.is_successful():
            return f"Success: {len(self.data)} rows generated"
        elif self.error:
            return f"Failed: {self.error}"
        else:
            return f"Status: {self.status.value}"


class BaseCalculation(ABC):
    """Abstract base class for all calculations."""

    def __init__(self, name: str):
        self.name = name
        self.status = CalculationStatus.PENDING
        self.result: Optional[CalculationResult] = None

    @abstractmethod
    def validate_inputs(self) -> tuple[bool, str]:
        """
        Validate inputs before calculation.

        Returns:
            Tuple of (is_valid, error_message)
        """
        pass

    @abstractmethod
    def execute(self, **kwargs) -> CalculationResult:
        """
        Execute the calculation.

        Returns:
            CalculationResult object
        """
        pass

    def get_progress(self) -> float:
        """Get calculation progress (0-100)."""
        if self.status == CalculationStatus.COMPLETED:
            return 100.0
        elif self.status == CalculationStatus.RUNNING:
            return 50.0
        return 0.0