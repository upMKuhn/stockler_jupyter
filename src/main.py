import pandas as pd
from models import train_price_ai as modeling


def main_train():
    data_set_files = [
        'symbol__TSLA_2018-08-02_2019-04-30',
        'symbol__DB_2018-08-02_2019-05-02',
        'symbol__FB_2018-08-02_2019-05-02',
    ]
    data_sets = {file: pd.read_csv(f'../data/{file}.csv') for file in data_set_files}
    for data in data_sets.values():
        data.index = pd.to_datetime(data.index)
        symbol = list(data_sets.keys())[0]
        data_sets[symbol] = data[['datetime', 'close', 'high', 'low', 'open', 'volume']]
        print(symbol)
    trainer = modeling.PriceAiTrainer(data_sets)
    model = trainer.build_model()
    model.summary()
    history = trainer.train(model, epochs=300)


def main_test():
    data_set_files = [
        'symbol__TSLA_2018-08-02_2019-04-30',
        'symbol__XBIO_2018-08-02_2019-04-30'
    ]
    data_sets = {file: pd.read_csv(f'../data/{file}.csv', index_col=0, parse_dates=True) for file in data_set_files}
    data = data_sets[data_set_files[0]]
    symbol = list(data_sets.keys())[0]
    print(symbol)
    model = modeling.PriceAiModel('../models/price_ai/.mdl_wts.hdf5')
    predictions = model.evaluate(data[:-40])
    print(predictions)


main_train()
