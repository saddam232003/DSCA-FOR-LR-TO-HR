def feature_volume_analysis(feature_sizes, model, dataset):
    results = {}
    for size in feature_sizes:
        model.set_feature_dimension(size)
        acc = model.train_and_evaluate(dataset)
        results[size] = acc
    return results