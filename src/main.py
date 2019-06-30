from pprint import pprint

from utils.stockler_influx import get_stockler_data_client


def main():
    client = get_stockler_data_client()
    result = client.query(""" 
       SELECT mean("average_price") AS mean_average_price FROM stockler.autogen.stocks 
       WHERE "symbol"='XBIO' GROUP BY time(30m) FILL(none)
    """.strip())

    client.close()
    return result


