import sqlite3
import pandas as pd


def prepare_df_for_db(dfs_final):
    for field_name, df_fin in dfs_final.items():
        new_col_names = [f"{field_name}_{col}" for col in df_fin.columns]
        col_rename = dict(zip(df_fin.columns, new_col_names))
        df_fin.rename(columns=col_rename, inplace=True)

    df_db = pd.concat(dfs_final.values(), axis=1)
    # print(df_db)

    return df_db


def write_to_db(df, year, out_file):
    table_name = f"year_{year}"

    # Create a database or connect to one
    conn = sqlite3.connect(out_file)

    # Create cursor
    c = conn.cursor()

    # Check if table exists
    c.execute(f"""SELECT count(name) FROM sqlite_master WHERE type='table' AND name='{table_name}'""")
    if c.fetchone()[0] == 1:
        print(f"Table {table_name} exists.")
        return

    # create table in db
    print(f"Creating table {table_name} in the database...")
    df.to_sql(name=table_name, con=conn)

    # Commit changes
    conn.commit()

    # Close connection
    conn.close()
