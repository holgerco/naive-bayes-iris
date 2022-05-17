from naive_bayes import NaiveBayes

if __name__ == '__main__':
    nb = NaiveBayes(
        data_headers=[
            'SepalLengthCm input',
            'SepalWidthCm input',
            'PetalLengthCm input',
            'PetalWidthCm input'
        ],
        target_headers=[
            'Iris-setosa target',
            'Iris-versicolor target',
            'Iris-virginica target'
        ]
    )
    nb.load_data('iris.csv')
    # nb.max_minx_normalization_data()
    nb.data_separation(0.8, 0.2)
    nb.calculate_estimators()
    nb.test_naive_bayes_model()
    nb.show_result()
