from Project.models.regression.train import train
from Project.models.regression.explain import explain
from Project.models.regression.evaluate import eval
from Project.models.regression.infer import infer
from Project.db.init_db import init_tables

# Pipeline
def run_regression_training_pipeline():
    init_tables()
    X,y=train()
    eval(X,y)
    explain()
    infer()
  



if __name__ == "__main__":
    run_regression_training_pipeline()
