
import pandas as pd
import psycopg2
from sqlalchemy import create_engine, text, types
import time

# Function to create the PostgreSQL table and insert data
def write_to_postgres(df, table_name, connection_params):
    """
    Write DataFrame to PostgreSQL database
    
    Parameters:
        df: pandas DataFrame to save
        table_name: name of the table to create
        connection_params: dictionary with connection parameters
    """
    # Create connection string
    conn_string = f"postgresql://{connection_params['user']}:{connection_params['password']}@{connection_params['host']}:{connection_params['port']}/{connection_params['dbname']}"
    
    try:
        # Connect to PostgreSQL using SQLAlchemy (for pandas to_sql)
        print(f"Connecting to PostgreSQL database: {connection_params['dbname']}...")
        engine = create_engine(conn_string)
        
        # Drop table if it exists
        print(f"Dropping existing table {table_name} if it exists...")
        with engine.connect() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {table_name};"))
            conn.commit()
        
        # Start timing
        start_time = time.time()
        
        # Write DataFrame to PostgreSQL
        print(f"Writing {len(df)} rows to table {table_name}...")
        
        # Determine appropriate SQL types for datetime columns
        dtypes = {}
        for col in df.select_dtypes(include=['datetime64']).columns:
            dtypes[col] = types.TIMESTAMP  # Use SQLAlchemy type objects
            
        # Write data to PostgreSQL
        df.to_sql(
            name=table_name,
            con=engine,
            if_exists='replace',
            index=False,
            chunksize=1000,
            dtype=dtypes
        )
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        print(f"Data successfully written to PostgreSQL in {elapsed_time:.2f} seconds")
        
        # Create primary key and indexes for better performance
        print("Creating primary key and indexes...")
        with engine.connect() as conn:
            conn.execute(text(f"ALTER TABLE {table_name} ADD PRIMARY KEY (id);"))
            
            # Create indexes on commonly used columns
            for col in ['category', 'subcategory', 'country', 'order_date']:
                if col in df.columns:
                    conn.execute(text(f"CREATE INDEX idx_{table_name}_{col.lower()} ON {table_name} ({col});"))
            conn.commit()
            
        print(f"Table {table_name} is ready for querying")
        return True
        
    except Exception as e:
        print(f"Error writing to PostgreSQL: {str(e)}")
        return False

