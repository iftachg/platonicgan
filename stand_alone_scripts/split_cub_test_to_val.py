'''
Goal of this script is to split the test set into validation and test sets
'''
from sklearn.model_selection import train_test_split
import os

base_path = '/media/john/D/projects/platonicgan/datasets/CUB/lists/'
origin_path = os.path.join(base_path, 'test.txt')
target_test_path = os.path.join(base_path, 'new_test.txt')
target_val_path = os.path.join(base_path, 'new_val.txt')

with open(origin_path, 'r') as f:
    file_names = list(f.readlines())

new_test, new_val = train_test_split(file_names, test_size=0.5)

a = 5

with open(target_test_path, 'w') as f:
    f.writelines(new_test)

with open(target_val_path, 'w') as f:
    f.writelines(new_val)
