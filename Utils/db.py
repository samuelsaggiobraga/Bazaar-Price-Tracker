import os
import json
from psycopg2.extras import RealDictCursor, execute_values
from psycopg2.pool import SimpleConnectionPool
from dotenv import load_dotenv
from pathlib import Path

# Load variables from the .env file into the system environment
load_dotenv()

_pool = None


def get_pool():
    global _pool
    if _pool is None:
        _pool = SimpleConnectionPool(1, 20, os.getenv("DATABASE_URL"))
    return _pool


def setup_timescaledb():
    # Initialize connection variable outside the try block for safe rollback in the except block
    connection = None

    try:
        # 1. Connect to the database
        connection = get_pool().getconn()
        cursor = connection.cursor()

        # 2. Query the information_schema to check if our tables exist
        check_query = """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name IN ('bazaar_item', 'bazaar_item_history');
        """
        cursor.execute(check_query)

        # Extract the table names from the fetched tuples
        existing_tables = [row[0] for row in cursor.fetchall()]

        # 3. Check the results and create tables/hypertables if missing
        if (
            "bazaar_item" in existing_tables
            and "bazaar_item_history" in existing_tables
        ):
            print(
                "✅ Tables 'bazaar_item' and 'bazaar_item_history' already exist. Moving on."
            )
        else:
            print("⚠️ Tables not found (or partially missing). Creating them now...")

            # Standard relational tables
            # NOTE: Double quotes are used to force PostgreSQL to respect camelCase
            create_tables_query = """
                CREATE TABLE IF NOT EXISTS bazaar_item (
                    id TEXT PRIMARY KEY
                );

                CREATE TABLE IF NOT EXISTS bazaar_item_history (
                    "itemId" TEXT REFERENCES bazaar_item(id) NOT NULL,
                    "timestamp" TIMESTAMPTZ NOT NULL,
                    "buy" REAL,
                    "sell" REAL,
                    "buyVolume" BIGINT,
                    "sellVolume" BIGINT,
                    "maxBuy" REAL,
                    "minBuy" REAL,
                    "maxSell" REAL,
                    "minSell" REAL,
                    "buyMovingWeek" BIGINT,
                    "sellMovingWeek" BIGINT,
                    PRIMARY KEY ("itemId", "timestamp")
                );
            """
            cursor.execute(create_tables_query)

            # TimescaleDB Hypertable Conversion
            # Note: Using 'timestamp' as the time partitioning column.
            print(
                "⏳ Converting 'bazaar_item_history' into a TimescaleDB hypertable..."
            )
            hypertable_query = """
                SELECT create_hypertable(
                    'bazaar_item_history', 
                    'timestamp', 
                    if_not_exists => TRUE
                );
            """
            cursor.execute(hypertable_query)

            # 4. Commit the transaction so changes are saved to the database
            connection.commit()
            print("✅ Tables and Hypertable created successfully.")

            cursor.close()

    except Exception as error:
        print(f"❌ Error connecting to or configuring TimescaleDB: {error}")

        # Roll back any partial, incomplete changes if an error occurs mid-transaction
        if connection:
            connection.rollback()
            print("↩️ Database transaction rolled back due to error.")

    finally:
        # Ensure the connection is ALWAYS closed, even if an exception is thrown
        if connection:
            get_pool().putconn(connection)
            print("🔌 Database connection safely returned to pool.")


def drop_timescaledb_tables():
    connection = None

    try:
        # Connect to the database
        connection = get_pool().getconn()
        cursor = connection.cursor()

        print("⚠️ Attempting to drop tables...")

        # Using CASCADE ensures that foreign key constraints don't block the drop
        # Dropping a hypertable (bazaar_item_history) automatically drops all its chunks
        drop_query = """
            DROP TABLE IF EXISTS bazaar_item_history CASCADE;
            DROP TABLE IF EXISTS bazaar_item CASCADE;
        """

        cursor.execute(drop_query)

        # Commit the transaction to apply the drops
        connection.commit()
        print("✅ Tables (and any associated hypertables/chunks) dropped successfully.")

        cursor.close()

    except Exception as error:
        print(f"❌ Error dropping tables: {error}")

        # Roll back if something goes wrong
        if connection:
            connection.rollback()
            print("↩️ Database transaction rolled back.")

    finally:
        # Safely close the connection
        if connection:
            get_pool().putconn(connection)
            print("🔌 Database connection safely returned to pool.")


def populate_bazaar_items(item_list):
    """
    Inserts a list of item IDs into the bazaar_item table.
    Ignores duplicates if they already exist in the database.
    """
    connection = None

    try:
        # Connect to the database
        connection = get_pool().getconn()
        cursor = connection.cursor()

        print(f"📦 Attempting to insert {len(item_list)} items into 'bazaar_item'...")

        # The SQL query with ON CONFLICT to safely ignore existing primary keys
        insert_query = """
            INSERT INTO bazaar_item (id) 
            VALUES %s 
            ON CONFLICT (id) DO NOTHING;
        """

        # We use list comprehension to format your list correctly
        data_to_insert = [(item,) for item in item_list]

        # Execute the batch insert
        execute_values(cursor, insert_query, data_to_insert)

        # Commit the transaction to save changes
        connection.commit()

        # To get the exact number of rows actually inserted (ignoring duplicates):
        print(
            f"✅ Successfully inserted {cursor.rowcount} NEW items into the bazaar_item."
        )

        cursor.close()

    except Exception as error:
        print(f"❌ Error inserting items: {error}")

        # Roll back if an error occurs
        if connection:
            connection.rollback()
            print("↩️ Database transaction rolled back.")

    finally:
        # Safely return the connection
        if connection:
            get_pool().putconn(connection)


def insert_item_history(item_id, history_list):
    """
    Takes an item_id and a list of history dictionaries from the API,
    and inserts them into the bazaar_item_history table using the
    exact camelCase column names.
    """
    if not history_list:
        print(f"⚠️ No history data provided for {item_id}.")
        return

    connection = None

    try:
        connection = get_pool().getconn()
        cursor = connection.cursor()

        print(
            f"📈 Attempting to insert {len(history_list)} history records for '{item_id}'..."
        )

        # The SQL query uses double quotes to enforce camelCase matching in PostgreSQL
        insert_query = """
            INSERT INTO bazaar_item_history (
                "itemId", "timestamp", "buy", "sell", "buyVolume", "sellVolume", 
                "maxBuy", "minBuy", "maxSell", "minSell", "buyMovingWeek", "sellMovingWeek"
            ) 
            VALUES %s 
            ON CONFLICT ("itemId", "timestamp") DO NOTHING;
        """

        # Build the list of tuples for execute_values
        data_to_insert = [
            (
                item_id,  # Injected from function parameter ("itemId")
                record.get("timestamp"),
                record.get("buy"),
                record.get("sell"),
                record.get("buyVolume"),
                record.get("sellVolume"),
                record.get("maxBuy"),
                record.get("minBuy"),
                record.get("maxSell"),
                record.get("minSell"),
                record.get("buyMovingWeek"),
                record.get("sellMovingWeek"),
            )
            for record in history_list
        ]

        # Execute the batch insert
        execute_values(cursor, insert_query, data_to_insert)

        # Commit the transaction
        connection.commit()

        print(
            f"✅ Successfully inserted {cursor.rowcount} NEW history records for '{item_id}'."
        )

        cursor.close()

    except Exception as error:
        print(f"❌ Error inserting history for {item_id}: {error}")
        if connection:
            connection.rollback()
            print("↩️ Database transaction rolled back.")

    finally:
        if connection:
            get_pool().putconn(connection)


def get_item_history(item_id, start_time=None, end_time=None, order_by="ASC"):
    """
    Fetches the history of a specific item.
    Excludes the 'itemId' from the returned dicts.
    Allows optional start_time and end_time filtering.
    """
    connection = None
    results = []

    try:
        connection = get_pool().getconn()
        cursor = connection.cursor(cursor_factory=RealDictCursor)

        # 1. Start with the base query and the required item_id parameter
        # We explicitly list the columns to exclude "itemId"
        query = """
            SELECT 
                "timestamp", "buy", "sell", "buyVolume", "sellVolume", 
                "maxBuy", "minBuy", "maxSell", "minSell", 
                "buyMovingWeek", "sellMovingWeek"
            FROM bazaar_item_history
            WHERE "itemId" = %s
        """
        params = [item_id]

        # 2. Dynamically append conditions if timestamps are provided
        if start_time:
            query += ' AND "timestamp" >= %s'
            params.append(start_time)

        if end_time:
            query += ' AND "timestamp" <= %s'
            params.append(end_time)

        # 3. Add an ORDER BY clause so the time-series data is returned chronologically or reversed-chronologically
        query += f' ORDER BY "timestamp" {order_by};'

        # 4. Execute and fetch
        cursor.execute(query, params)
        results = cursor.fetchall()

        print(f"✅ Fetched {len(results)} history records for '{item_id}'.")

        cursor.close()

    except Exception as error:
        print(f"❌ Error fetching history for {item_id}: {error}")

    finally:
        if connection:
            get_pool().putconn(connection)

    return results


def get_latest_timestamp(item_id):
    """
    Fetches the most recent (latest) timestamp available for a given item.
    Returns a timezone-aware Python datetime object, or None if no data exists.
    """
    connection = None
    latest_time = None

    try:
        connection = get_pool().getconn()
        cursor = connection.cursor()

        # Using MAX() to efficiently find the newest record
        # Keep the double quotes for camelCase/exact matching
        query = """
            SELECT MAX("timestamp") 
            FROM bazaar_item_history 
            WHERE "itemId" = %s;
        """

        cursor.execute(query, (item_id,))

        # fetchone() returns a tuple with one item, e.g., (datetime(...),)
        result = cursor.fetchone()

        # Check if we got a result and if the value inside isn't None
        if result and result[0]:
            latest_time = result[0]
            print(f"✅ The latest timestamp for '{item_id}' is: {latest_time}")
        else:
            print(f"⚠️ No history data found for '{item_id}'.")

        cursor.close()

    except Exception as error:
        print(f"❌ Error fetching latest timestamp for {item_id}: {error}")

    finally:
        if connection:
            get_pool().putconn(connection)

    return latest_time


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    file_path = os.path.join(
        root,
        "Backend",
        "all_bazaar_items.json",
    )
    with open(file_path, "r", encoding="utf-8") as file:
        # Load and parse the JSON content into a Python dictionary or list
        data = json.load(file)
        populate_bazaar_items(data)

    # setup_timescaledb()
    # drop_timescaledb_tables()
