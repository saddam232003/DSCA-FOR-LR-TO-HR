def component(model, dataset):
    configs = {
        'full': model,
        'no_spearman': model.remove('spearman'),
        'linear_only': model.replace('rbf', 'linear'),
        'no_batchnorm': model.remove('batchnorm')
    }
    results = {}
    for label, config_model in configs.items():
        acc = config_model.train_and_evaluate(dataset)
        results[label] = acc
    return results