import pandas as pd
from pendulum import datetime, parse

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import pickle
from pathlib import Path

from prefect import flow, task

@task
def read_data(path):
    df = pd.read_parquet(path)
    return df

@task
def prepare_features(df, categorical, train=True):
    # logger = get_run_logger()
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        print(f"The mean duration of training is {mean_duration}")
    else:
        print(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task
def train_model(df, categorical):

    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    print(f"The shape of X_train is {X_train.shape}")
    print(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    print(f"The MSE of training is: {mse}")
    return lr, dv

@task
def run_model(df, categorical, dv, lr):
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    print(f"The MSE of validation is: {mse}")
    return

@task
def get_paths(date):
    train_date = date.add(months=-2)
    val_date = date.add(months=-1)
    train_path = f'./data/fhv_tripdata_{train_date.strftime("%Y-%m")}.parquet'
    val_path = f'./data/fhv_tripdata_{val_date.strftime("%Y-%m")}.parquet'

    return train_path, val_path   

def save_model(model, dv, date, directory="models"):
    Path(directory).mkdir(parents=True, exist_ok=True)
    date_str = date.strftime("%Y-%m-%d")
    model_file_name = f"model-{date_str}.bin"
    dv_file_name = f"dv-{date_str}.bin"
    with open(f'{directory}/{model_file_name}', 'wb') as f_out:
        pickle.dump(model, f_out)
    with open(f'{directory}/{dv_file_name}', 'wb') as f_out:
        pickle.dump(dv, f_out)    

@flow
def main(date=None):
    date_obj = parse(date)
    train_path, val_path = get_paths(date_obj).result()
    print(train_path)
    print(val_path)

    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()
    save_model(lr, dv, date_obj)
    run_model(df_val_processed, categorical, dv, lr)

# main("2021-08-15")

from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule
from prefect.deployments import DeploymentSpec
from prefect.flow_runners import SubprocessFlowRunner

DeploymentSpec(
    flow=main,
    name="model_training",
    schedule=CronSchedule(cron="0 9 15 * *",
        timezone="Europe/London"),
    flow_runner=SubprocessFlowRunner(),
    tags=["ml"],
)