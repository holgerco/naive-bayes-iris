from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from pandas._typing import FilePath


class NaiveBayes:

    def __init__(self, *args, **kwargs):
        self.raw_data: pd.DataFrame | None = None
        self.attribute_names: List[str] = kwargs.get('data_headers', [])
        self.class_names: List[str] = kwargs.get('target_headers', [])
        self.train_data: Dict[str, pd.DataFrame] = {}
        self.test_data: Dict[str, pd.DataFrame] = {}
        self.data_variance: Dict[str, Dict[str, float]] = {}
        self.data_mean: Dict[str, Dict[str, float]] = {}
        self.errors: List[Tuple[int, str, str]] = []
        self._test_data_count = None

    def load_data(self, path: FilePath) -> None:
        data = pd.read_csv(path)  # load file
        self.raw_data = pd.DataFrame(data)  # create data structure

    def max_minx_normalization_data(self) -> None:
        # apply normalization techniques
        for column in self.raw_data.columns:
            if 'input' in column:
                self.raw_data[column] = (self.raw_data[column] - self.raw_data[column].min()) / (
                        self.raw_data[column].max() - self.raw_data[column].min())

    def data_separation(self, train_weights, test_weights) -> None:
        if train_weights + test_weights != 1:
            raise ValueError('sum of train and test weights must be 1')
        self._test_data_count = test_weights * len(self.raw_data)
        for header in self.class_names:
            _header_data: pd.DataFrame = self.raw_data[self.raw_data[header] == 1]
            self.train_data[header]: pd.DataFrame = _header_data.sample(frac=train_weights)
            self.test_data[header] = _header_data.drop(self.train_data[header].index)

    def calculate_estimators(self):
        for header in self.class_names:
            self.data_variance[header] = {}
            self.data_mean[header] = {}
            for attribute in self.attribute_names:
                self.data_variance[header][attribute] = self.train_data[header][attribute].var()
                self.data_mean[header][attribute] = self.train_data[header][attribute].mean()

    def calculate_class_probability(self, class_name):
        return len(self.raw_data[self.raw_data[class_name] == 1]) / len(self.raw_data)

    def calculate_naive_probability(self, x_i, class_name, attribute):
        mean = self.data_mean[class_name][attribute]
        variance = self.data_variance[class_name][attribute]
        return (1 / np.sqrt(variance)) * np.exp(
            -(1 / (2 * (variance ** 2))) * ((x_i - mean) ** 2)
        )

    def argmax(self, row):
        probabilities_value = 0
        probabilities_class = None
        for class_name in self.class_names:
            prob = self.calculate_class_probability(class_name=class_name)
            prob *= np.sqrt(1 / (2 * np.pi))
            for attribute in self.attribute_names:
                prob *= self.calculate_naive_probability(
                    x_i=row[attribute],
                    class_name=class_name,
                    attribute=attribute,
                )
            if probabilities_value < prob:
                probabilities_value = prob
                probabilities_class = class_name
        return probabilities_class

    def test_naive_bayes_model(self):
        for class_name in self.class_names:
            for index, row in self.test_data[class_name].iterrows():
                predicted_class = self.argmax(row)
                if predicted_class != class_name:
                    self.errors.append([index, class_name, predicted_class])

    def show_result(self):
        error_rate = len(self.errors) / self._test_data_count * 100
        print(F'Naive Bayes classification error rate: {error_rate:.2f}%')
        print(F'Naive Bayes classification accuracy rate: {100 - error_rate:.2f}%')
