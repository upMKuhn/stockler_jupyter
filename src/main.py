
import pandas as pd
from models import train_price_ai as modeling

def main():
    data_set_files = [
        'symbol__TSLA_2018-08-02_2019-04-30',
        'symbol__XBIO_2018-08-02_2019-04-30'
    ]
    validation_percent = 0.25
    data_sets = {file: pd.read_csv(f'../data/{file}.csv') for file in data_set_files}
    for data in data_sets.values():
        data.index = pd.to_datetime(data.index)
        symbol = list(data_sets.keys())[0]
        print(symbol)
        trainer = modeling.PriceAiTrainer(data_sets)
        model = trainer.build_model()
        history = trainer.train(model, symbol)

main()