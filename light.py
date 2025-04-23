import pandas as pd
from lightgbm import LGBMRegressor, Dataset
import lightgbm
from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np


df = pd.read_csv('pure_final_letter.csv')
df = df.dropna(subset=['LogP'])

lower_bound = df['LogP'].quantile(0.0035)
upper_bound = df['LogP'].quantile(0.9965)
df = df[(df['LogP'] >= lower_bound) & (df['LogP'] <= upper_bound)]

def is_valid_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

valid_mask = df['SMILES'].apply(is_valid_smiles)
df = df[valid_mask].copy()
df = df.drop(['ID'], axis=1)


def get_rdkit_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    descriptors = {}
    for desc_name, desc_func in Descriptors._descList:
        try:
            descriptors[desc_name] = desc_func(mol)
        except:
            descriptors[desc_name] = np.nan

    return descriptors


descriptors_data = df['SMILES'].apply(get_rdkit_descriptors).apply(pd.Series)
data = pd.concat([df, descriptors_data], axis=1)
data = data.dropna(axis=1, how='all')
data = data.dropna(subset=['LogP'])
data.to_csv('deskriptors.csv', index=False)
data = pd.read_csv('deskriptors.csv')
X_train = data.drop(['SMILES', 'LogP'], axis=1)
y_train = data['LogP']
train_data = Dataset(X_train, label=y_train)


model = LGBMRegressor(
    num_leaves=21,
    learning_rate=0.03,
    n_estimators=500000, #МОЖНО СОКРАТИТЬ ДО 100 000, ЧТОБЫ НЕ ЖДАТЬ ДОЛГО (качество незначительно просядет)
    objective='mae',
    metric='rmse',
    verbose=-1
)

model.fit(
    X_train,
    y_train,
    eval_metric='rmse',
    callbacks=[lightgbm.log_evaluation(period=100)]
)


model.booster_.save_model('lightgbm_logp_model.txt')


df_test = pd.read_csv('test_data.csv')
descriptors_data_test = df_test['SMILES'].apply(get_rdkit_descriptors).apply(pd.Series)
data_test = pd.concat([df_test, descriptors_data_test], axis=1)
T_test = data_test.drop(['SMILES', 'ID'], axis=1)
test_pred = model.predict(T_test)

submit_df = pd.DataFrame()
submit_df['ID'] = df_test['ID']
submit_df['LogP'] = test_pred
submit_df.to_csv('test_predict.csv', index=False)