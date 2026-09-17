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
from .methods import get_database_url
Base = declarative_base()

class Stocks(Base):
    __tablename__ = 'stocks'
    
    ticker = Column(String, nullable=False, primary_key=True)  # Ticker symbol
    conid = Column(Integer, nullable=False)  # Contract ID
    currency = Column(String(3), nullable=False)  # Currency code, e.g., USD
    listing_exchange = Column(String, nullable=False)  # Listing exchange
    country_code = Column(String(2), nullable=False)  # Country code, e.g., US
    name = Column(String, nullable=False)  # Name of the stock or ETF
    asset_class = Column(String, nullable=False)  # Asset class, e.g., STK
    group = Column(String)  # Group or category of the stock
    sector = Column(String)  # Sector of the stock
    sector_group = Column(String)  # Sector group of the stock
    type = Column(String, nullable=False)  # Type of asset, e.g., COMMON or ETF
    has_options = Column(Boolean, nullable=False)  # Indicates if options are available

# Define the Stock_Prices_1m table
class StockPrices30m(Base):
    __tablename__ = 'stock_prices_30m'
    
    price_id = Column(String, primary_key=True)
    datasource = Column(String, nullable=False) 
    ticker = Column(String, ForeignKey('stocks.ticker'), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)

# Define the Stock_Prices_5m table
class StockPrices1h(Base):
    __tablename__ = 'stock_prices_1h'
    
    price_id = Column(String, primary_key=True)
    datasource = Column(String, nullable=False) 
    ticker = Column(String, ForeignKey('stocks.ticker'), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)

# Define the Stock_Prices_1h table
class StockPrices2h(Base):
    __tablename__ = 'stock_prices_2h'
    
    price_id = Column(String, primary_key=True)
    datasource = Column(String, nullable=False) 
    ticker = Column(String, ForeignKey('stocks.ticker'), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)

# Define the Stock_Prices_1d table
class StockPrices1D(Base):
    __tablename__ = 'stock_prices_1d'
    
    price_id = Column(String, primary_key=True)
    datasource = Column(String, nullable=False) 
    ticker = Column(String, ForeignKey('stocks.ticker'), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)

class StockOptunaResults30m(Base):
    __tablename__ = 'stock_optuna_results_30m'

    
    trial_id = Column(String, primary_key=True)  # Unique trial identifier
    ml_model = Column(String, nullable=False)  # Machine learning model used
    trial_num = Column(Integer, nullable=False)  # Trial number
    trial_rewards = Column(Float, nullable=False)  # Trial rewards
    trial_metric = Column(String, nullable=False)  # Trial metric (e.g., precision)
    trial_num_preds = Column(Float, nullable=False)  # Number of predictions
    trial_ave_profit = Column(Float, nullable=False)  # Average profit
    trial_ave_loss = Column(Float, nullable=False)  # Average loss
    trial_params = Column(JSON, nullable=False)  # Trial parameters as a string or JSON
    study_datetime = Column(DateTime, nullable=False)  # Trial start time
    ticker = Column(String, ForeignKey('stocks.ticker'), nullable=False)  # Ticker symbol
    source = Column(String, nullable=False)  # Data source/API used
    timeframe = Column(String, nullable=False)  # Timeframe (e.g., '30m')
    trial_start = Column(DateTime, nullable=False)  # Trial start time
    trial_end = Column(DateTime, nullable=False)  # Trial end time

    

def create_tables() -> None:
    """
    Creates all defined SQLAlchemy model tables in the PostgreSQL database.

    Initializes the database schema by creating tables for all models that inherit
    from SQLAlchemy Base. Uses environment variables for database connection parameters.

    Notes:
        - Requires properly configured environment variables for database connection
        - Creates tables only if they don't already exist
        - Must be run after all model classes are defined
        - Uses SQLAlchemy declarative base metadata for table creation
        - Safe to run multiple times (idempotent operation)

    Raises:
        SQLAlchemyError: If table creation fails or database connection issues occur
        EnvironmentError: If required environment variables are missing

    Example:
        # After defining your models:
        >>> create_tables()
        # Tables are created in the database
    """
    conn_string =get_database_url()
    engine = create_engine(conn_string)
    Base.metadata.create_all(engine)

if __name__ == '__main__':
    create_tables()
