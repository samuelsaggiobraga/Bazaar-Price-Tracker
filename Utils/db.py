# CONNECT TO DATABASE
# CREATE SCHEMA IF IT DOESNT EXISTS
# WRITE QUERIES AND EXPORT

import os
import psycopg2
from dotenv import load_dotenv

# Load variables from the .env file into the system environment
load_dotenv()

# CONSIDER USING A CONNECTION POOL TO REUSE CONNECTION

try:
    connection = psycopg2.connect(os.getenv("DATABASE_URL"))

    # Create a cursor object to execute database operations
    cursor = connection.cursor()

    # Execute a simple query to test the connection and check the TimescaleDB version
    cursor.execute("SELECT extversion FROM pg_extension WHERE extname = 'timescaledb';")
    version = cursor.fetchone()

    if version:
        print(f"Successfully connected! TimescaleDB extension version: {version[0]}")
    else:
        print(
            "Connected to PostgreSQL, but TimescaleDB extension is not active in this database."
        )

    # Close communication paths
    cursor.close()
    connection.close()

except Exception as error:
    print(f"Error connecting to TimescaleDB: {error}")
