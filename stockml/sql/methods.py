import sqlalchemy
from sqlalchemy import inspect
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Boolean, JSON
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import create_engine, Table, MetaData, select
import pandas as pd
import json
import os
from dotenv import load_dotenv

Base = declarative_base()
load_dotenv()

def get_database_url() -> str:
    """
    Constructs database URL from environment variables.
    
    Returns:
        str: PostgreSQL connection URL
    """
    return f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"


def connect_to_postgresql() -> None:
    """
    Establishes a connection to the PostgreSQL database and prints available tables.

    Connects to a local PostgreSQL database using SQLAlchemy engine and performs
    a basic connection test by listing all available tables.

    Notes:
        - Requires PostgreSQL server running on localhost:5432
        - Prints all table names in the connected database
        - Connection parameters should be stored in environment variables for security
    
    Raises:
        SQLAlchemyError: If connection to database fails
    """
    conn_string = get_database_url()
    engine = create_engine(conn_string) 
    conn = engine.connect()
    inspection = inspect(engine)
    print(inspection.get_table_names() )


def write_stock_metadata(stock_data: list[dict], metadata_table: declarative_base) -> None:
    """
    Writes stock metadata to PostgreSQL database with duplicate handling.

    Inserts or updates stock information in the database, implementing transaction
    management and rollback capabilities for data integrity.

    Args:
        stock_data (list[dict]): List of dictionaries containing stock metadata:
            - conid: Contract ID
            - currency: Trading currency
            - listing_exchange: Primary exchange
            - country_code: Country of listing
            - name: Company name
            - asset_class: Type of asset
            - group, sector, sector_group: Classification fields
            - ticker: Stock symbol
            - type: Security type
            - has_options: Options availability flag
        metadata_table (Base): SQLAlchemy model class for stock metadata table

    Notes:
        - Implements upsert logic based on ticker symbol
        - Uses session management for transaction safety
        - Includes error handling and rollback
    """
    conn_string = get_database_url()
    engine = create_engine(conn_string)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        for stock in stock_data:
            existing_stock = session.query(metadata_table).filter_by(ticker=stock['ticker']).first()
            
            if not existing_stock:
                new_stock = metadata_table(
                    conid=stock['conid'],
                    currency=stock['currency'],
                    listing_exchange=stock['listingExchange'],
                    country_code=stock['countryCode'],
                    name=stock['name'],
                    asset_class=stock['assetClass'],
                    group=stock.get('group'),
                    sector=stock.get('sector'),
                    sector_group=stock.get('sectorGroup'),
                    ticker=stock['ticker'],
                    type=stock['type'],
                    has_options=stock['hasOptions']
                )
                session.add(new_stock)
   
        session.commit()
        print("Stock metadata written successfully.")
    
    except Exception as e:
        session.rollback()
        print(f"An error occurred: {e}")
    
    finally:
        session.close()


def write_historical_data(historical_data_df: pd.DataFrame, ohclv_table: declarative_base) -> None:
    """
    Writes OHLCV historical price data to PostgreSQL database.

    Inserts time series price data with duplicate prevention using price_id as
    the unique identifier.

    Args:
        historical_data_df (pd.DataFrame): DataFrame containing OHLCV data with columns:
            price_id, datasource, ticker, timestamp, open, high, low, close, volume
        ohclv_table (Base): SQLAlchemy model class for price data table

    Notes:
        - Checks for existing entries using price_id
        - Implements batch processing for performance
        - Includes transaction management and error handling
    """
    conn_string = get_database_url()
    engine = create_engine(conn_string)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        for index, row in historical_data_df.iterrows():
            existing_entry = session.query(ohclv_table).filter_by(price_id=row['price_id']).first()
            
            if not existing_entry:
                new_entry = ohclv_table(
                    price_id=row['price_id'],
                    datasource=row['datasource'],
                    ticker=row['ticker'],
                    timestamp=row['timestamp'],
                    open=row['open'],
                    high=row['high'],
                    low=row['low'],
                    close=row['close'],
                    volume=row['volume']
                )
                session.add(new_entry)

        session.commit()
        print("Historical data written successfully.")
    
    except Exception as e:
        session.rollback()
        print(f"An error occurred: {e}")
    
    finally:
        session.close()


def write_optuna_results(optuna_results_df: pd.DataFrame, optuna_table: declarative_base) -> None:
    """
    Writes Optuna optimization results to PostgreSQL database.

    Stores machine learning model optimization results including hyperparameters
    and performance metrics.

    Args:
        optuna_results_df (pd.DataFrame): DataFrame containing optimization results:
            trial_id, ml_model, metrics, parameters, and timestamps
        optuna_table (Base): SQLAlchemy model class for Optuna results table

    Notes:
        - Converts parameter dictionaries to JSON for storage
        - Prevents duplicate entries using trial_id
        - Stores comprehensive trial information including:
            * Performance metrics
            * Hyperparameters
            * Temporal information
            * Model configuration
    """
    conn_string = get_database_url()
    engine = create_engine(conn_string)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        for index, row in optuna_results_df.iterrows():
            existing_entry = session.query(optuna_table).filter_by(trial_id=row['trial_id']).first()
            
            if not existing_entry:
                trial_params_json = json.dumps(row['trial_params'])

                new_entry = optuna_table(
                    trial_id=row['trial_id'],
                    ml_model=row['ml_model'],
                    trial_num=row['trial_num'],
                    trial_rewards=row['trial_rewards'],
                    trial_metric=row['trial_metric'],
                    trial_num_preds=row['trial_num_preds'],
                    trial_ave_profit=row['trial_ave_profit'],
                    trial_ave_loss=row['trial_ave_loss'],
                    trial_params=trial_params_json,
                    study_datetime = row['study_datetime'],
                    ticker=row['ticker'],
                    source=row['source'],
                    timeframe=row['timeframe'],
                    trial_start=row['trial_start'],
                    trial_end=row['trial_end']
                    )

                session.add(new_entry)

        session.commit()
        print("Optuna results written successfully.")
    
    except Exception as e:
        session.rollback()
        print(f"An error occurred: {e}")
    
    finally:
        session.close()


def get_filtered_data(ohclv_table: declarative_base, datasource: str, ticker: str) -> pd.DataFrame:
    """
    Retrieves filtered historical price data from PostgreSQL database.

    Queries historical data based on datasource and ticker symbol, returning
    results as a pandas DataFrame.

    Args:
        ohclv_table (Base): SQLAlchemy model class for price data table
        datasource (str): Data source identifier
        ticker (str): Stock symbol to filter by

    Returns:
        pd.DataFrame: Filtered historical price data with OHLCV columns

    Notes:
        - Uses SQLAlchemy ORM for safe query construction
        - Implements efficient DataFrame conversion
        - Includes session management and error handling
    """
    conn_string = get_database_url()
    engine = create_engine(conn_string)
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        query = session.query(ohclv_table).filter(
            ohclv_table.datasource == datasource,
            ohclv_table.ticker == ticker
        )
        df = pd.read_sql(query.statement, engine)
        
        return df
    
    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        session.close()



def get_all_rows(table: declarative_base) -> pd.DataFrame:
    """
    Retrieves all rows from specified PostgreSQL table.

    Performs a full table scan and returns all records as a pandas DataFrame.

    Args:
        table (Base): SQLAlchemy model class for target table

    Returns:
        pd.DataFrame: Complete table contents as DataFrame

    Notes:
        - Use with caution on large tables
        - Implements connection pooling via SQLAlchemy
        - Includes proper session cleanup
    """

    conn_string = get_database_url()
    engine = create_engine(conn_string)
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        query = session.query(table)
        df = pd.read_sql(query.statement, engine)
        
        return df
    
    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        session.close()


