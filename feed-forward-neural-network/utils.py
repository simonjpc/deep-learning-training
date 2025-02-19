def batcher(dataset, batch_size=32):
    x_train, y_train = dataset

    dataset_batches = []
    cnt = 0
    while True:
        start = cnt * batch_size
        end = (cnt + 1) * batch_size
        single_input_batch = x_train[start:end]
        single_output_batch = y_train[start:end]
        dataset_batches.append((single_input_batch, single_output_batch))
        if start >= len(x_train) or end >= len(x_train):
            break
        cnt += 1
    return dataset_batches
