import numpy as np
import pandas as pd
import requests
from utils.files_util import save_files

url = "https://publicapi.traffy.in.th/share/teamchadchart/search"

def load_data():
  # A GET request to the API
  response = requests.get(url)

  # Print the response
  response_json = response.json()
  print(response_json)

  df = pd.DataFrame(response_json['results'])
  df.name = "fondue"

  save_files([df])