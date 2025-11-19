import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor

def load_data():
  # Load from file
  sal_df = pd.read_csv('Salary_Data.csv')
  sal_df.columns = sal_df.columns.str.lower().str.replace(' ', '_')

  # Clean data
  sal_df = sal_df.dropna(subset=['age'])
  sal_df = sal_df.dropna(subset=['salary'])
  sal_df.education_level = sal_df.education_level.fillna('na')
  sal_df.education_level = sal_df.education_level.str.lower()
  sal_df.education_level = sal_df.education_level.str.split(' ', n=1).str[0]
  sal_df.education_level = sal_df.education_level.str.replace("'", "")

  # Split data into training sets
  full_train_df, test_df = train_test_split(sal_df, test_size=0.2, random_state=55)
  train_df, val_df = train_test_split(full_train_df, test_size=0.25, random_state=55)
  full_train_y = full_train_df.salary.values
  test_y = test_df.salary.values
  train_y = train_df.salary.values
  val_y = val_df.salary.values
  del full_train_df['salary']
  del test_df['salary']
  del train_df['salary']
  del val_df['salary']

  return train_df, train_y, val_df, val_y, test_df, test_y

def dict_and_vectorize(tr_df, v_df):
    dv = DictVectorizer(sparse=False)
    train_dict = tr_df.to_dict(orient='records')
    val_dict = v_df.to_dict(orient='records')

    train_X = dv.fit_transform(train_dict)
    val_X = dv.transform(val_dict)

    return train_X, val_X, dv

def train_rforest(tr_df, tr_y, val_df, val_y, n_e, m_d):
    train_X, val_X, dv = dict_and_vectorize(tr_df, val_df)

    rf = RandomForestRegressor(n_estimators=n_e, max_depth=m_d, random_state=55)
    rf.fit(train_X, tr_y)

    return dv, rf

def pickle_model(dv, model):
  with open('midterm_model.bin', 'wb') as f_out:
    pickle.dump((dv, model), f_out)

def run():
  print('Loading data...')
  train_df, train_y, val_df, val_y, test_df, test_y = load_data()

  print('Training model...')
  # See `midterm.ipynb` for model parameter tuning
  dv, model = train_rforest(train_df, train_y, val_df, val_y, 40, 26)

  print('Exporting model...')
  pickle_model(dv, model)
  print('Done!')

if __name__ == "__main__":
    run()