import model_lib.dataset as diff


dataset = diff.UnconditionalDataset(["./data/pilot_vessel"])


for i in range(len(dataset)):
    dict = dataset[i]

    print(dict["audio"].shape, " ", dict["spectrogram"])
    break