import pandas as pd
import click

from src.data_preprocessing.eda import (
    heatmap_corr_plot,
    loan_status_count,
    loan_status_grade,
)

from .modeling.model import predict_dataset, train_dataset
from .modeling.utils import (
    feature_importance_plot,
    optimal_num_tree_graph,
    cross_validation_report,
    xgboost_tree_plot,
)
from .data_preprocessing.preprocessing import preprocessing, remove_redundant_target


@click.command()
@click.option(
    "-p", "--path-dataset", prompt=False, default="data/accepted_2007_to_2018Q4.csv"
)
def main(path_dataset):

    df = pd.read_csv(path_dataset)
    heatmap_corr_plot(df)
    _df = remove_redundant_target(df)
    loan_status_count(_df)
    loan_status_grade(_df)
    df, X_train, X_test, y_train, y_test = preprocessing(df)

    ###############################################################
    model, results = train_dataset(X_train, y_train, X_test, y_test)
    _, acc = predict_dataset(model, X_test, y_test)
    print(f"acc is {acc}")
    optimal_num_tree_graph(model, results)
    params = cross_validation_report(X_train, y_train)
    xgboost_tree_plot(X_train, y_train, params)
    feature_importance_plot(df, model)


if __name__ == "__main__":
    main()
