import sqlite3
import pandas as pd

conn = sqlite3.connect("noshow.db")  # or your actual path
tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
print("Tables found in database:")
print(tables)

conn.close()
