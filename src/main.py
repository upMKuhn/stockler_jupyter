import pandas as pd
from models import train_price_ai as modeling


def main_train():
    data_set_files = [
        'symbol__TSLA_2018-08-02_2019-04-30',
        'symbol__XBIO_2018-08-02_2019-04-30'
    ]
    data_sets = {file: pd.read_csv(f'../data/{file}.csv') for file in data_set_files}
    for data in data_sets.values():
        data.index = pd.to_datetime(data.index)
        symbol = list(data_sets.keys())[0]
        print(symbol)
        trainer = modeling.PriceAiTrainer(data_sets)
        model = trainer.build_model()
        history = trainer.train(model, symbol)


def main_test():
    data_set_files = [
        'symbol__TSLA_2018-08-02_2019-04-30',
        'symbol__XBIO_2018-08-02_2019-04-30'
    ]
    data_sets = {file: pd.read_csv(f'../data/{file}.csv', index_col=0, parse_dates=True) for file in data_set_files}
    data = data_sets[data_set_files[0]]
    symbol = list(data_sets.keys())[0]
    print(symbol)
    model = modeling.PriceAiModel('../models/price_ai__2019-07-08T00:32:03.628635')
    predictions = model.evaluate(data[:-40])
    print(predictions)

main_test()
