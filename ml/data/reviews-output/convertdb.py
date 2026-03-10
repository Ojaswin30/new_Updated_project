import sqlite3
import pandas as pd

# connect to database
conn = sqlite3.connect("D:\\github\\git repositories\\new_Updated_project\\ml\\data\\reviews-output\\product_ranking.sqlite")

# get all tables
tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)

print(tables)

# export each table
for table in tables['name']:
    
    # skip internal sqlite table
    if table == "sqlite_sequence":
        continue
        
    df = pd.read_sql(f"SELECT * FROM {table}", conn)
    
    df.to_csv(f"{table}.csv", index=False)
    
    print(f"{table}.csv exported successfully")

conn.close()