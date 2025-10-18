"""
Input validation utilities
"""

import tkinter as tk
from tkinter import messagebox


class Validators:
    """Collection of input validation methods."""

    @staticmethod
    def validate_float(value):
        """Validate that input is a float."""
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False

    @staticmethod
    def validate_positive_float(value):
        """Validate that input is a positive float."""
        try:
            num = float(value)
            return num >= 0
        except (ValueError, TypeError):
            return False

    @staticmethod
    def validate_integer(value):
        """Validate that input is an integer."""
        try:
            int(value)
            return True
        except (ValueError, TypeError):
            return False

    @staticmethod
    def validate_positive_integer(value):
        """Validate that input is a positive integer."""
        try:
            num = int(value)
            return num > 0
        except (ValueError, TypeError):
            return False

    @staticmethod
    def validate_percentage(value):
        """Validate that input is a percentage (0-100)."""
        try:
            num = float(value)
            return 0 <= num <= 100
        except (ValueError, TypeError):
            return False

    @staticmethod
    def validate_temperature(value, unit='K'):
        """Validate temperature input."""
        try:
            temp = float(value)
            if unit == 'K':
                return temp > 0  # Absolute zero
            elif unit == 'C':
                return temp >= -273.15  # Absolute zero in Celsius
            return True
        except (ValueError, TypeError):
            return False

    @staticmethod
    def validate_composition_range(start, end, step):
        """Validate composition range parameters."""
        try:
            start_val = float(start)
            end_val = float(end)
            step_val = float(step)

            if step_val <= 0:
                return False, "Step must be positive"

            if start_val > end_val:
                return False, "Start value cannot be greater than end value"

            return True, "Valid"

        except (ValueError, TypeError):
            return False, "Invalid number format"

    @staticmethod
    def show_validation_error(field_name, issue):
        """Show a standardized validation error message."""
        messagebox.showerror(
            "Validation Error",
            f"Invalid input for {field_name}:\n{issue}"
        )


def create_float_validator(widget):
    """Create a validator that only allows float input."""

    def validate(action, value_if_allowed):
        if action == '1':  # Insertion
            try:
                float(value_if_allowed)
                return True
            except ValueError:
                return False
        return True

    widget.configure(validate='key', validatecommand=(widget.register(validate), '%d', '%P'))


def create_positive_float_validator(widget):
    """Create a validator that only allows positive float input."""

    def validate(action, value_if_allowed):
        if action == '1':  # Insertion
            try:
                num = float(value_if_allowed)
                return num >= 0
            except ValueError:
                return False
        return True

    widget.configure(validate='key', validatecommand=(widget.register(validate), '%d', '%P'))


def create_integer_validator(widget):
    """Create a validator that only allows integer input."""

    def validate(action, value_if_allowed):
        if action == '1':  # Insertion
            try:
                int(value_if_allowed)
                return True
            except ValueError:
                return False
        return True

    widget.configure(validate='key', validatecommand=(widget.register(validate), '%d', '%P'))