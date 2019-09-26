#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import requests
import json
headers = {'Content-Type': 'application/json'}

root_url = 'http://localhost:8080/classificationbox'

def get(url, headers, payload=None):
    resp = requests.get(url, data=payload, headers=headers)
    print(resp)
    respObj = json.loads(resp.text)
    return respObj, resp.status_code

def post(url, headers, payload=None):
    resp = requests.post(url, data=payload, headers=headers)
    print(resp.status_code)
    print(resp.text)
    respObj = json.loads(resp.text)
    return respObj, resp.status_code

def delete(url, headers):
    resp = requests.delete(url, headers=headers)
    print(resp.status_code)


# ### Create a model

# In[ ]:


model_url = f'{root_url}/models'
model_payload = {
	"id": "iris",
	"name": "iris",
	"options": {},
	"classes": [
		"Iris Setosa",
		"Iris Versicolor",
		"Iris Virginica"
	]
}

resp, code = post(model_url, headers, payload=json.dumps(model_payload))


# ### delete/list model

# In[ ]:


delete(f'{model_url}/iris',headers)


# In[ ]:


get(f'{model_url}',headers)


# ### teach

# In[ ]:


import csv
examples = []
train_payload = {'examples':examples}
with open('iris.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        example = {}
        example['class'] = row['species']
        example['inputs'] = []
        for key in row.keys():
            if key != 'species':
                item = {}
                item['key'] = key
                item['type'] = 'number'
                item['value'] = str(row[key])
                example['inputs'].append(item)
        examples.append(example)        
print(json.dumps(train_payload, indent=2))


# In[ ]:


teach_url = f'{model_url}/iris/teach-multi'
resp, code = post(teach_url, headers, payload=json.dumps(train_payload))


# ### Predict

# In[ ]:


predict_url = f'{model_url}/iris/predict'
predict_payload = {"limit": 10}
predict_payload['inputs'] = examples[40]['inputs']
print(examples[40])

resp, code = post(predict_url, headers, payload=json.dumps(predict_payload))


# ### Model Statistics

# In[ ]:


stats_url = f'{model_url}/iris/stats'
resp, code = get(stats_url, headers)
print(json.dumps(resp,indent=2))

