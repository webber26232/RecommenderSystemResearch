from dask.distributed import Client
import dask.dataframe as dd


def parser(file_path):
    import os
    file_name = os.path.split(file_path)[1]
    return int(file_name.split('_')[1].split('.')[0])


client = Client(n_workers=2, threads_per_worker=1,
                processes=False, memory_limit='6.5GB')

print(client.dashboard_link)

ddf = dd.read_csv(
    'C:/Users/YuJun/Documents/local_python/recommender_system/data/training_set/*.txt',
    include_path_column='itemID',
    skiprows=1,
    header=None,
    usecols=[0, 1],
    names=['userID', 'rating'],
    converters={'itemID': parser})


def get_sample_user(user_column):
    user_counts = user_column.value_counts().compute()
    return user_counts.index.to_series().sample(1000, weights=user_counts.values)


print('filtering and saving.......................')
ddf[ddf['userID'].isin(get_sample_user(ddf['userID']))].repartition(npartitions=50).to_csv('C:/Users/YuJun/Documents/local_python/recommender_system/data/Netflix_Filtered_*.csv', index=False)

