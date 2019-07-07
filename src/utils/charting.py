import settings
import numpy as np
from pandas import DataFrame


def candle_coloring(data: DataFrame, color_a=None, color_b=None):
    color_a = color_a or settings.INCREASING_COLOR
    color_b = color_b or settings.DECREASING_COLOR
    colors = []
    for i in range(len(data)):
        if i != 0:
            if data[i] > data[i - 1]:
                colors.append(color_a)
            else:
                colors.append(color_b)
        else:
            colors.append(color_b)
    return colors


def get_figure_with_sub_chart(has_rangeselector=False, y_axis_range=None):
    """
        Creates Figure with 2 charts or 2 yaxis and range selector
        this should be the starting point ;)
    """
    layout = dict(
        plot_bgcolor='rgb(250, 250, 250)',
        xaxis=dict(rangeslider=dict(visible=has_rangeselector)),
        yaxis2=dict(domain=[0, 0.2], showticklabels=False, autorange=True),
        yaxis=dict(domain=[0.2, 0.8], autorange=(not bool(y_axis_range))),
        legend=dict(orientation='h', y=0.9, x=0.3, yanchor='bottom'),
        margin=dict(t=40, b=40, r=40, l=40)
    )
    if y_axis_range:
        layout['yaxis']['range'] = y_axis_range

    if has_rangeselector:
        rangeselector = dict(
            visibe=has_rangeselector,
            x=0, y=0.9,
            bgcolor='rgba(150, 200, 250, 0.4)',
            font=dict(size=13),
            buttons=list([
                dict(count=1,
                     label='reset',
                     step='all'),
                dict(count=1,
                     label='1yr',
                     step='year',
                     stepmode='backward'),
                dict(count=3,
                     label='3 mo',
                     step='month',
                     stepmode='backward'),
                dict(count=1,
                     label='1 mo',
                     step='month',
                     stepmode='backward'),
                dict(step='all')
            ])
        )
        layout['xaxis']['rangeslider'] = rangeselector
    return dict(data=[], layout=layout)


def add_candle_chart(figure, open, low, high, close, x_data, y_axis='y', name='symbol'):
    candle_sticks = dict(
        type='candlestick',
        open=open,
        high=high,
        low=low,
        close=close,
        x=x_data,
        yaxis=y_axis,
        name=name,
        increasing=dict(line=dict(color=settings.INCREASING_COLOR)),
        decreasing=dict(line=dict(color=settings.DECREASING_COLOR))
    )
    figure['data'].append(candle_sticks)


def add_scatter_chart(figure, y_data, x_data, y_axis='y', name='Scatter', showlegend=True, color="#ccc"):
    figure['data'].append(dict(x=x_data, y=y_data, type='scatter', yaxis=y_axis,
                               line=dict(width=1),
                               marker=dict(color=color), hoverinfo='none',
                               legendgroup=name, showlegend=showlegend, name=name))


def add_bar_chart(figure, x_data, y_data, yaxis='y', colors=None, name='Bar Chart'):
    colors = colors or candle_coloring(y_data)
    figure['data'].append(dict(
        x=x_data, y=y_data,
        type='bar', yaxis=yaxis, name=name,
        marker=dict(color=colors)
    ))


def calculate_moving_average(interval_name, df: DataFrame, window_size=10):
    interval = df[interval_name]
    window = np.ones(int(window_size)) / float(window_size)
    avg = np.convolve(interval, window, 'same')
    slice_first = int(round(window_size / 2))
    slice_last = int(round(window_size / 2)) * -1
    return DataFrame(
        avg[slice_first:slice_last],
        index=list(df.index)[slice_first:slice_last],
        columns=['average']
    ).dropna()


def calculate_bollinger_bands(data_name, df: DataFrame, window_size=20, num_of_std=5):
    """ Thanks :) https://plot.ly/~jackp/17421/plotly-candlestick-chart-in-python/#/"""
    price = df[data_name]
    rolling_mean = price.rolling(window=window_size).mean()
    rolling_std = price.rolling(window=window_size).std()
    upper_band = rolling_mean + (rolling_std * num_of_std)
    lower_band = rolling_mean - (rolling_std * num_of_std)
    columns = ['avg', 'upper', 'lower']
    df = DataFrame([], index=df.index, columns=columns)
    df['upper'] = upper_band
    df['lower'] = lower_band
    df['avg'] = rolling_mean
    return df.dropna()
