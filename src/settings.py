import os

INFLUX_HOST = os.environ.get('INFLUX_HOST', 'influx')
INFLUX_PORT = int(os.environ.get('INFLUX_PORT', '8086'))
INFLUX_DB = os.environ.get('INFLUX_DB', 'stockler')
INFLUX_USER = os.environ.get('INFLUX_USER', 'root')
INFLUX_PASSWORD = os.environ.get('INFLUX_PASSWORD', 'root')

INFLUX_ARGS = {
    'host': INFLUX_HOST,
    'port': INFLUX_PORT,
    'username': INFLUX_USER,
    'password': INFLUX_PASSWORD
}


INCREASING_COLOR = '#68A456'
DECREASING_COLOR = '#D1314A'