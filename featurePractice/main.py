# 201835506 임동혁
# EX_featuretools

import pandas as pd
import numpy as np
import featuretools as ft
from woodwork.logical_types import Categorical, PostalCode

# Bring data
clients = pd.read_csv('data/clients.csv', parse_dates=['joined'])
loans = pd.read_csv('data/loans.csv', parse_dates=['loan_start', 'loan_end'])
payments = pd.read_csv('data/payments.csv', parse_dates=['payment_date'])

# Create new entityset
es = ft.EntitySet(id='clients')

#Create an entity from the client dataframe
es = es.add_dataframe(dataframe_name='clients', dataframe=clients,
                      index='client_id', time_index='joined')

#Create an entity from the loans dataframe
es = es.add_dataframe(dataframe_name='loans', dataframe=loans,
                      logical_types={'repaid': Categorical},
                      index='loan_id',
                      time_index='loan_start')

es = es.add_dataframe(dataframe_name='payments',
                      dataframe=payments,
                      logical_types={'missed': Categorical},
                      make_index=True,
                      index='payment_id',
                      time_index='payment_date')

#aggregation
#Group loans by client id and calculate total of  loans
stats = loans.groupby('client_id')['loan_amount'].agg(['sum'])
stats.columns = ['total_loan_amount']

# Merge with the clients dataframe
stats = clients.merge(stats, left_on='client_id', right_index=True, how='left')
print(stats.head(10))

# Add the relationship to the entity set
es = es.add_relationship('clients', 'client_id', 'loans', 'client_id')
es = es.add_relationship('loans','loan_id', 'payments','loan_id')

print(es)
primitives = ft.list_primitives()
pd.options.display.max_colwidth = 100
primitives[primitives['type'] == 'aggregation'].head(10)
features, feature_names = ft.dfs(entityset = es, target_dataframe_name = 'clients',
                                 agg_primitives=['sum'],
                                 trans_primitives=[ 'month'])
print(pd.DataFrame(features['MONTH(joined)'].head()))
