from glob import glob
from sklearn.model_selection import train_test_split
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--configuration_file', '-cf', required=True, help="Represent the path for configuration file for the dataset split configuration")

args = parser.parse_args()

classes = {
    '02691156': 0,
    '02828884': 1,
    '02933112': 2,
    '02958343': 3,
    '03001627': 4,
    '03211117': 5,
    '03636649': 6,
    '03691459': 7,
    '04090263': 8,
    '04256520': 9,
    '04379243': 10,
    '04401088': 11,
    '04530566': 12
}

if __name__ == "__main__":
    with open(args.configuration_file, 'r') as config_file:
        config_parameter = json.load(config_file)

    datalist = glob(config_parameter['start_folder'] + "/*/*/*/" + config_parameter['image_name'])
    traindata, testdata = train_test_split(datalist, test_size=config_parameter['test_percentage'])
    traindata, valdata = train_test_split(datalist, test_size=config_parameter['validation_percentage'])

    with open(config_parameter['save_folder'] + 'train.txt', 'w') as train_file:
        for t in traindata:
            train_file.write(t + ',' + str(classes[t.replace(config_parameter['start_folder'], '').split('/')[0]]) + '\n')

    with open(config_parameter['save_folder'] + 'validation.txt', 'w') as val_file:
        for t in valdata:
            val_file.write(t + ',' + str(classes[t.replace(config_parameter['start_folder'], '').split('/')[0]]) + '\n')

    with open(config_parameter['save_folder'] + 'test.txt',  'w') as test_file:
        for t in testdata:
            test_file.write(t + ',' + str(classes[t.replace(config_parameter['start_folder'], '').split('/')[0]]) + '\n')