import pickle
import numpy as np

hyperparams = {
    'dataset': ['COLLAB'],
    # ,'COLLAB', 'IMDBBINARY', 'IMDBMULTI', 'NCI1','PROTEINS', 'PTC']#,'REDDITBINARY': [], 'REDDITMULTI5K': [],}
    'final_dropout': [0, 0.5],
    'hidden_layers': [64],
    'batch_size': 128,
    'epochs': 700,
    'fold': list(range(10)),
    'features': 'spectral'
}

max_acc = 0
best_hyperparams = None
best_std = 0
for dataset_name in hyperparams['dataset']:
    for final_dropout in hyperparams['final_dropout']:
        for hidden_layers in hyperparams['hidden_layers']:
            test_accs = []
            for fold in range(10):
                with open(
                        f"./results/{dataset_name}/FD={str(final_dropout)}-HL={hidden_layers}-F={hyperparams['features']}-FOLD={fold}",
                        "rb") as fp:
                    accs = pickle.load(fp)
                    test_accs.append(accs)

            for epoch in range(0, hyperparams['epochs']):
                accs = []
                for fold in range(10):
                    accs += [test_accs[fold][epoch]]
                accs = np.array(accs)
                avg_acc = accs.mean()
                std_acc = accs.std()
                if avg_acc > max_acc:
                    max_acc = avg_acc
                    best_hyperparams = (dataset_name, final_dropout, hidden_layers, epoch)
                    best_std= std_acc

print(f'validation accuracy: {max_acc}±{best_std}, params: {best_hyperparams}')


