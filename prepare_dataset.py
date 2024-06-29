# download cifar100 dataset with tfds
import os

path = '/root/autodl-tmp'
import pickle
import numpy as np
from PIL import Image
import os

def load_cifar_file(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file, encoding='bytes')
    return data

def save_images(data, class_names, root_dir):
    for i, (image_array, label) in enumerate(zip(data[b'data'], data[b'fine_labels'])):
        image = image_array.reshape(3, 32, 32).transpose(1, 2, 0)
        class_dir = os.path.join(root_dir, class_names[label])
        os.makedirs(class_dir, exist_ok=True)
        image_path = os.path.join(class_dir, f'image_{i}.jpg')  # Changed extension to .jpg
        Image.fromarray(image).save(image_path, 'JPEG')  # Specified format as 'JPEG'

def main():
    dataset_dir = os.path.join(path, 'cifar-100-python')
    train_data = load_cifar_file(os.path.join(dataset_dir, 'train'))
    test_data = load_cifar_file(os.path.join(dataset_dir, 'test'))
    meta = load_cifar_file(os.path.join(dataset_dir, 'meta'))
    class_names = [name.decode('utf-8') for name in meta[b'fine_label_names']]

    train_root_dir = os.path.join(path, 'cifar-100/train')
    test_root_dir = os.path.join(path, 'cifar-100/test')

    save_images(train_data, class_names, train_root_dir)
    save_images(test_data, class_names, test_root_dir)

if __name__ == '__main__':
    main()