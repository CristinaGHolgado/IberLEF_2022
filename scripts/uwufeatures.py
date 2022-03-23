# Sample file to demonstrate how to use UMUTextStats from an API
import requests
import json, time
import tempfile
import pandas as pd
import numpy as np
from tqdm import tqdm
# @var sample_text String 
sample_text = 'Vamos a probar con un texto un poco más largo'


# @var umutextstats_endpoint String The end point
umutextstats_endpoint = 'https://umuteam.inf.um.es/umutextstats/api/'

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


#################################################
# login ######################################
######################################


# First step. Login at the system
# @todo. Put your credentials
# @var login_request_data Object
login_request_data = {
    'email': 'amansinha091@gmail.com',
    'password': 'iambeer9'
}


# Make login request
r = requests.post (umutextstats_endpoint + 'login', json = login_request_data, verify = False)
#print(f"login time {r.status_code}")
if 200 != r.status_code:
    raise ValueError ('Login failed')


# Store the authentication token
auth_token = r.json ()['data']['token']

print("Logged into server ....")
#################################################################################

def run_wuw(text, auth_token):

    # @var text_request_data Object
    text_request_data = {
        'source-provider': 'text',
        'content': sample_text,
        'format': 'csv',
        'model': 'umutextstats',
        'umutextstats-config': 'default.xml'
    }


    # Do request
    r = requests.post (
        umutextstats_endpoint + 'stats.csv', 
        json = text_request_data, 
        verify = False, 
        headers = {
            'Authorization': auth_token,
            'Content-type': 'text/html; charset=UTF-8'
        }
    )

    #print(f'sending request time : {r.status_code}')
    if 200 != r.status_code:
        raise ValueError ('Request failed')


    # Get the file
    # We strip the response, as we return some extra bytes to the progress bar
    response = json.loads (r.text[r.text.find ('{'):])
    file = response['file']

    r = requests.get (
        umutextstats_endpoint + file,
        verify = False, 
        headers = {
            'Authorization': auth_token
        }
    )


    # Get the response
    data = r.text


    # Extract the values
    rows = [x.split (',') for x in data.split ('\n')[1:-1]]


    # Get columns to build a dataframe
    columns = [x for x in data.split ('\n')[0].split (',')][:-1]


    # Get final dataframe
    features = [row[:-1] for row in rows]
    features = list (np.float_ (features))

    return features, columns

#################################################################################


df = pd.read_csv('~/IberLEF_2022/data/development_test.csv')


feats = []
times = []
t1 = time.time()
for i,sen in tqdm(enumerate(df.tweet)) :
    #print(sen)
    feat, col = run_wuw(sen, auth_token)
    feats.append(feat[0])
    t2 = time.time()
    times.append(t2 - t1)
    t1 = t2
    if i == 1:
        break

#print(feats, col)
features = pd.DataFrame(feats, columns = col)


features.to_csv('demo.csv', sep='\t')
print('testing csv saved')
print(f'Average extraction time : {np.mean(times):.4f} seconds')

# Here you can store the results in a CSV or combine several dataframes
#print (features)
