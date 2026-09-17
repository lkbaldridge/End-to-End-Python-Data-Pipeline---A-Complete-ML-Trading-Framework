import requests
import subprocess
import pytz
from datetime import datetime, timedelta, timezone
import pandas as pd
import json

def pull_ib_contract_list(instrument: str = "ETF.EQ.US", type: str = "OPT_VOLUME_MOST_ACTIVE") -> pd.DataFrame | None:
    """
    Retrieves a list of contracts from Interactive Brokers based on specified criteria.

    Makes a POST request to IB's scanner API to fetch contracts matching the specified
    instrument type and filtering criteria.

    Args:
        instrument (str, optional): Instrument category to scan. Defaults to "ETF.EQ.US".
        type (str, optional): Scanner type for filtering. Defaults to "OPT_VOLUME_MOST_ACTIVE".

    Returns:
        pd.DataFrame | None: DataFrame containing contract details with columns:
            - con_id: Contract identifier
            - company_name: Name of the company
            - scan_data: Scanner-specific data
            - last_price: Latest trading price
            - listing_exchange: Primary exchange
            - sec_type: Security type
            Returns None if request fails.

    Notes:
        - Requires active IB Gateway or TWS connection on port 5000
        - Uses SSL verification bypass (not recommended for production)
        - Filters for minimum option volume of 5
    """

    base_url = "https://localhost:5000/v1/api/"
    request_url = f"{base_url}/iserver/scanner/run"

    json_content = {
        "instrument": instrument,
        "location": f"{instrument}.MAJOR",
        "type": type,
        "filter": [{
                    "code": "optVolume",
                    "value": 5
        }]
    }


    req_post = requests.post(url=request_url, json=json_content, verify=False)

    if req_post.status_code == 200:
        js = req_post.json()
        df = pd.DataFrame(js['contracts'])
        df.set_index('symbol', inplace=True)

        return df[['con_id', 'company_name', 'scan_data', 'last_price', 'listing_exchange', 'sec_type']] 
    
    else:
        print(f'Error: {req_post.status_code}')
        return None


def pull_ib_stock_data(con_id: str, ticker: str, start_now: bool = True, 
                      period: str = '1y', bar_size: str = '30mins') -> pd.DataFrame:
    """
    Retrieves historical stock data from Interactive Brokers for a specific contract.

    Fetches OHLCV data with specified granularity and time period, converting timestamps
    to Pacific timezone.

    Args:
        con_id (str): IB contract identifier
        ticker (str): Stock symbol
        start_now (bool, optional): If True, uses current time as start. Defaults to True.
        period (str, optional): Historical data period. Defaults to '1y'.
        bar_size (str, optional): Candle size for data. Defaults to '30mins'.

    Returns:
        pd.DataFrame: Historical price data with columns:
            price_id, datasource, ticker, timestamp, open, high, low, close, volume

    Notes:
        - Automatically adjusts timestamps for Pacific timezone
        - Includes outside regular trading hours data
        - Creates unique price_id combining ticker, source, and timestamp
    """

    if start_now == True:
        base_url =  'https://localhost:5000/v1/api'
        request_url = f"{base_url}/hmds/history?conid={con_id}&period={period}&bar={bar_size}&outsideRth=true"

        los_angeles_tz = pytz.timezone('America/Los_Angeles')
        now = datetime.now(tz=los_angeles_tz)
        now += timedelta(hours=7)
        formatted_date = now.strftime('%Y%m%d-%H:%M:%S')

    else:
        startTime = start_now
        base_url =  'https://localhost:5000/v1/api'
        request_url = f"{base_url}/hmds/history?conid={con_id}&exchange=SMART&period=1y&bar=30min&outsideRth=true&startTime={startTime}"
    
    req_get = requests.get(url=request_url,verify=False)
    data = json.loads(req_get.content)
    data_df = pd.DataFrame(data['data'])

    data_df['time'] = data_df['t'].apply(unix_to_pacific)
    data_df['ticker'] = ticker
    data_df['datasource'] = 'ibapi'
    data_df['price_id'] = data_df.apply(lambda row: f"{row['ticker']} {row['datasource']} {str(row['time'])}", axis=1)

    data_df.columns = ['ts', 'open', 'close', 'high', 'low', 'volume', 'timestamp', 'ticker', 'datasource', 'price_id']

    return data_df[['price_id', 'datasource', 'ticker', 'timestamp', 'open', 'high', 'low', 'close', 'volume']]


def contract_search(symbol: str) -> bytes:
    """
    Searches for contract information using IB's security definition search.

    Args:
        symbol (str): Stock symbol to search for

    Returns:
        bytes: Raw JSON response containing contract search results

    Notes:
        - Limited to STK (stock) security type
        - Requires active IB Gateway connection
        - Returns raw response content for flexible parsing
    """
    base_url = "https://localhost:5000/v1/api/"
    endpoint = "iserver/secdef/search"
    json_body = {"symbol" : symbol, "secType": "STK", "name": False}
    contract_req = requests.post(url=base_url+endpoint, verify=False, json=json_body)
    
    return contract_req._content


def contract_details(con_id: str) -> list[dict]:
    """
    Retrieves detailed contract information from IB's security definition service.

    Args:
        con_id (str): IB contract identifier

    Returns:
        list[dict]: List of dictionaries containing contract details including:
            con_id, currency, listingExchange, countryCode, name, assetClass,
            group, sector, sectorGroup, ticker, type, hasOptions

    Notes:
        - Returns empty list if no data found
        - Filters response to specific fields of interest
        - Requires active IB Gateway connection
    """
    base_url = "https://localhost:5000/v1/api/"
    request_url = f"{base_url}/trsrv/secdef?con_ids={con_id}"
    req_get = requests.get(url=request_url, verify=False)
    data = json.loads(req_get.content)

    desired_fields = ['con_id', 'currency', 'listingExchange', 'countryCode', 'name', 'assetClass', 'group', 'sector', 'sectorGroup', 'ticker', 'type', 'hasOptions']
    
    if 'secdef' in data:
        extracted_data = [{field: item.get(field) for field in desired_fields}
                         for item in data['secdef']]
        return extracted_data
    
    else:
        return []


def get_contract_id(contract_details: bytes) -> str | None:
    """
    Extracts contract ID from IB contract details response.

    Args:
        contract_details (bytes): Raw contract details response from IB API

    Returns:
        str | None: Contract ID if found, None otherwise

    Notes:
        - Handles UTF-8 decoding of response
        - Expects JSON format in bytes response
        - Prints extracted contract ID for verification
    """
    if contract_details is not None:
        contract_details_str = contract_details.decode('utf-8')
        contract_details_dict = json.loads(contract_details_str)
        con_id = contract_details_dict[0]['con_id']
        print(f"Contract ID: {con_id}")

        return con_id
    
    else:
        print("No contract details found.")


def unix_to_pacific(unix_timestamp: float) -> str:
    """
    Converts Unix timestamp (milliseconds) to Pacific timezone datetime string.

    Args:
        unix_timestamp (float): Unix timestamp in milliseconds

    Returns:
        str: Formatted datetime string in Pacific timezone (YYYY-MM-DD HH:MM:SS)

    Notes:
        - Uses timezone-aware objects throughout conversion process
        - Handles conversion from milliseconds to seconds automatically
        - Returns consistent format regardless of daylight savings
    """
    unix_seconds = unix_timestamp / 1000
    utc_datetime = datetime.fromtimestamp(unix_seconds, timezone.utc)
    pacific_tz = pytz.timezone('US/Pacific')
    pacific_datetime = utc_datetime.astimezone(pacific_tz)

    return pacific_datetime.strftime('%Y-%m-%d %H:%M:%S')