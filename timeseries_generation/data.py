import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

"""
Dataset for Heatrow weather data downloaded from
https://www.metoffice.gov.uk/pub/data/weather/uk/climate/stationdata/heathrowdata.txt
downloaded on July 2023
"""

class WeatherDataset(Dataset):
    def __init__(self, max_length=None):
        columns = ['Year','Month','tmax','tmin','af','rain','sun']
        rows = []
        with open("sequence_generation_timeseries/metoffice.gov.uk_pub_data_weather_uk_climate_stationdata_heathrowdata.txt") as f:
            for i,line in enumerate(f):
                if i >= 7:
                    row_values = line.replace("#","").replace("*","").split()[:7]
                    for k in [4, 6]:
                        if row_values[k] == "---":
                            row_values[k] = "Nan"
                    rows.append(row_values)

        self.df = pd.DataFrame(columns=columns,data=rows).astype(np.float32).iloc[:-6].dropna()

    def __len__(self):
        return self.df.shape[0] - 24 - 3
    
        
    def __getitem__(self, i, include_year=False):
        df_slice = self.df.iloc[i:i+24+3]
        if not include_year:
            df_slice = df_slice.iloc[:,1:]
        return torch.from_numpy(df_slice.iloc[:,:3].values)

if __name__ == "__main__":
    dataset = WeatherDataset()
    x = dataset[0]
    print(x)
