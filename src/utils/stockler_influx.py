from datetime import datetime

from influxdb import InfluxDBClient, DataFrameClient
from pandas import DataFrame

import settings as settings
import re

stockler_client = None
stockler_data_client = None


def get_stockler_main_client():
    global stockler_client
    if not stockler_client:
        stockler_client = InfluxDBClient(database='stockler', **settings.INFLUX_ARGS)
    return stockler_client


def get_stockler_data_client() -> DataFrameClient:
    global stockler_data_client
    if not stockler_data_client:
        stockler_data_client = DataFrameClient(database='stockler', **settings.INFLUX_ARGS)
    return stockler_data_client


def select_from_stocks(query: str, symbol_regex='', epoch='1m', min_date: datetime = None, max_date: datetime = None)\
        -> DataFrame:
    """ Builds a select query to cut the boilerplate"""
    client = get_stockler_data_client()
    group_by = f"GROUP BY time({epoch})" if epoch else ''
    date_filter = ''
    if max_date:
        timezone = 'Z'
        date_filter = f"AND time >= \'{max_date.isoformat()}{timezone}\'"
    if min_date:
        timezone = 'Z'
        date_filter += f" AND time >= \'{min_date.isoformat()}{timezone}\'"

    query = f""" 
           SELECT 
            {query} 
           FROM stockler.autogen.stocks 
           WHERE "symbol" =~ {symbol_regex} {date_filter} {group_by} FILL(null)
        """.strip()
    query = re.sub("^\s*", '', query, flags=re.MULTILINE)
    print(query)

    result = client.query(query, epoch=epoch)
    if len(result):
        return result['stocks']
    return result
