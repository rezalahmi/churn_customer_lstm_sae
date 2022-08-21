# This is a sample Python script.
from flower_pollination import flower_pollination_optimizer
from prepareData import split_train_test
from lstm_sae import lstm_sae

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dataset_path = 'dataset/selected_column.csv'

    Xtrain, Xtest = split_train_test(0.8, dataset_path)
    optimizer = flower_pollination_optimizer(Xtrain, Xtest, 64, 32, 16)
    opt_data = optimizer.jfs()
    print(opt_data['sf'])
    print(opt_data['c'])
    print(opt_data['nf'])





# See PyCharm help at https://www.jetbrains.com/help/pycharm/
