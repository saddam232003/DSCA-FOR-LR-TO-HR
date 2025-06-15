def class_specific_performance(model, dataset):
    class_results = {}
    for cls in dataset.classes:
        cls_data = dataset.filter_by_class(cls)
        acc = model.evaluate(cls_data)
        class_results[cls] = acc
    return class_results