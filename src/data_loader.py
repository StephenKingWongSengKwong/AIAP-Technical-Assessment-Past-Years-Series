"""
Data loading and database connection handling
"""

import sqlite3
import pandas as pd
from typing import Optional


class DataLoader:
    def __init__(self, db_path: str):
        """Initialize the data loader with database path.

        Args:
            db_path (str): Path to SQLite database
        """
        self.db_path = db_path
        self.connection: Optional[sqlite3.Connection] = None

    def connect(self) -> None:
        """Establish database connection."""
        self.connection = sqlite3.connect(self.db_path)

    def disconnect(self) -> None:
        """Close database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None

    def load_data(self, query: str) -> pd.DataFrame:
        """Load data from database using SQL query.

        Args:
            query (str): SQL query to execute

        Returns:
            pd.DataFrame: Retrieved data
        """
        try:
            if not self.connection:
                self.connect()
            return pd.read_sql_query(query, self.connection)
        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()