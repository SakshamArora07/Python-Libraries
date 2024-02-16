
EPOCHS = 10
BATCH_SIZE_TRAIN = 16
BATCH_SIZE_TEST = 10
LR = 0.0001
LOG_INTERVAL = 100
RANDOM_SEED = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")
TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Resize([224, 224]),
])

CLASSES = {'dogs': 0, 'cats': 1}

torch.manual_seed(RANDOM_SEED)

train_dataset = MyDataset(get_df('cats_and_dogs_filtered/train/'), CLASSES, TRANSFORM)
test_dataset = MyDataset(get_df('cats_and_dogs_filtered/validation/'), CLASSES, TRANSFORM)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE_TEST, shuffle=True)

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)
def get_df(path, classes=['dogs', 'cats']):
    paths = pd.DataFrame({'class': [], 'path': []})
    for c in classes:
        df = pd.DataFrame({
            'class': c,
            'path': glob.glob(path + c + '/*')
        })

        paths = pd.concat([paths, df])

    paths.reset_index(inplace=False)

    return paths
