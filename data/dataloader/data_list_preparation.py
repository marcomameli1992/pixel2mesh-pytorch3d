from glob import glob
from sklearn.model_selection import train_test_split
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--configuration_file', '-cf', required=True, help="Represent the path for configuration file for the dataset split configuration")

args = parser.parse_args()

if __name__ == "__main__":
    with open(args.configuration_file, 'r') as config_file:
        config_parameter = json.load(config_file)

    datalist = glob(config_parameter['start_folder'] + "/*/*/*/" + config_parameter['image_name'])
    traindata, testdata = train_test_split(datalist, test_size=config_parameter['test_percentage'])
    traindata, valdata = train_test_split(datalist, test_size=config_parameter['validation_percentage'])

    with open(config_parameter['save_folder'] + 'train.txt', 'w') as train_file:
        for t in traindata:
            train_file.write(t + '\n')

    with open(config_parameter['save_folder'] + 'validation.txt', 'w') as val_file:
        for t in valdata:
            val_file.write(t + '\n')

    with open(config_parameter['save_folder'] + 'test.txt',  'w') as test_file:
        for t in testdata:
            test_file.write(t + '\n')