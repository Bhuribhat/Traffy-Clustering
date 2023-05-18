import pandas as pd
import numpy as np
from utils.files_util import save_files, load_files
from scipy import stats

def clean_data(ti, **context):
  df = load_files(['fondue'])[0]

  # Drop some columns
  drop_col = [
    'ticket_id', 'description', 'after_photo', 'photo_url', 
    'star', 'count_reopen', 'last_activity', 'org'
  ]
  df.drop(drop_col, axis=1, inplace=True)

  # Convert timestamp to datetime
  df['timestamp'] = pd.to_datetime(df['timestamp']).apply(lambda x: x.date())

  # split coordinate column into separate latitude and longitude columns
  df[['latitude', 'longitude']] = df['coords'].apply(lambda x: x.strip('\"[]\"')).str.split(', ', expand=True)
  df['latitude'] = df['latitude'].apply(lambda x: x.strip("\'"))
  df['longitude'] = df['longitude'].apply(lambda x: x.strip("\'"))

  # drop the original coordinate column
  df.drop(['coords'], axis=1, inplace=True)

  df['province'] = df['address'].str.split(' ').str[-3]

  # rename province in Bangkok
  df['province'] = df['province'].apply(lambda x: "กรุงเทพมหานคร" if x == "จังหวัดกรุงเทพมหานคร" else x)

  df['district'] = df['address'].str.split(' ').str[-4]

  # split 'type' in each row to list datatype
  df['type'] = df['problem_type_abdul'].apply(lambda x: x.strip('\"[]').replace("\'", "").split(", "))

  # add new column to count 'type' length
  df['type_count'] = df['type'].apply(lambda x: len(x))

  # convert coord column to float
  df = df.astype({"latitude": float, "longitude": float})

  # Calculate z-scores for latitude and longitude columns
  z_lat  = np.abs(stats.zscore(df['latitude']))
  z_long = np.abs(stats.zscore(df['longitude']))

  # Define threshold value
  threshold = 3

  # Remove outliers from the dataframe
  df = df[(z_lat < threshold) & (z_long < threshold)]

  # explode list of type to multiple rows
  df_exploded = df.explode('type')
  df_exploded.drop('type_count', axis=1, inplace=True)

  # cleaning empty string
  df_exploded['type'] = df_exploded['type'].str.strip("\'\"")
  df_exploded.drop(df_exploded[df_exploded['type'] == ''].index, inplace=True)

  # filter data based on parameter in body
  filter_criteria = []
  try:
    filter_criteria = context['dag_run'].conf['filter']
    print(filter_criteria)
    df_exploded = df_exploded[df_exploded['type'].isin(filter_criteria)]
  except Exception as e:
    print("error lol")
  finally:
    df_exploded.name = "cleaned_fondue"
    df.name = "data_cleaned_by_length"
    save_files([df_exploded, df])

