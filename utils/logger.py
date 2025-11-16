"""
Logging utilities for the application
"""

import logging
import tkinter as tk
from tkinter import ttk


class ApplicationLogger:
    """Custom logger for the alloy design application."""

    def __init__(self, log_widget=None):
        self.log_widget = log_widget
        self.setup_logging()

    def setup_logging(self):
        """Setup basic logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('alloy_design.log'),
                logging.StreamHandler()
            ]
        )

    def log(self, message, level='info'):
        """Log a message to both file and UI widget."""
        # Log to file
        if level == 'info':
            logging.info(message)
        elif level == 'warning':
            logging.warning(message)
        elif level == 'error':
            logging.error(message)
        elif level == 'debug':
            logging.debug(message)

        # Log to UI widget if available
        if self.log_widget:
            self.log_to_widget(message, level)

    def log_to_widget(self, message, level):
        """Log message to UI text widget."""
        try:
            # Color coding based on level
            colors = {
                'info': 'black',
                'warning': 'orange',
                'error': 'red',
                'debug': 'gray'
            }

            color = colors.get(level, 'black')

            # Insert with color
            self.log_widget.insert(tk.END, message + '\n', level)
            self.log_widget.tag_config(level, foreground=color)
            self.log_widget.see(tk.END)

        except Exception as e:
            # Fallback if widget logging fails
            print(f"Widget logging failed: {e}")


def create_log_frame(parent):
    """Create a standardized log frame with scrollbar."""
    log_frame = ttk.LabelFrame(parent, text="Application Log", padding="5")

    # Create scrollbar
    scrollbar = ttk.Scrollbar(log_frame)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # Create text widget
    log_text = tk.Text(
        log_frame,
        height=10,
        yscrollcommand=scrollbar.set,
        wrap=tk.WORD,
        font=('Courier New', 9)
    )
    log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Configure scrollbar
    scrollbar.config(command=log_text.yview)

    return log_frame, log_text
