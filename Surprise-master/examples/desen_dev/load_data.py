from surprise import Dataset, Reader
from dask.distributed import Client
import pandas as pd
import dask.dataframe as dd


client = Client(n_workers=3, threads_per_worker=1,
                processes=False, memory_limit='5GB')

print(client.dashboard_link)

ddf = dd.read_csv('Netflix_Filtered_2005-01-01_*.csv')


df = ddf[['userID', 'itemID', 'rating']].compute()


# A reader is still needed but only the rating_scale param is requiered.
reader = Reader(rating_scale=(1, 5))
# The columns must correspond to user id, item id and ratings (in that order).
data = Dataset.load_from_df(df, reader)
data.split(5)  # data can now be used normally
