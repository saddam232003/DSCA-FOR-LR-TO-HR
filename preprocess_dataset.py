def preprocess_dataset(images, resolution=(24, 24), add_noise=True):
    preprocessed = []
    for img in images:
        img_resized = resize_image(img, resolution)
        if add_noise:
            img_resized += np.random.normal(0, 0.01, img_resized.shape)
        img_normalized = (img_resized - img_resized.mean()) / 255.0
        preprocessed.append(img_normalized)
    return np.array(preprocessed)