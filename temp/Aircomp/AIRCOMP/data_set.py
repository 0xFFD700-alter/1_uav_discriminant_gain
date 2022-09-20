import os
from PIL import Image
import numpy as np


def gen_txt(dir, train):
    folder_level_list = os.listdir(dir)
    folder_level_list.sort()
    for folder_level in folder_level_list:
        for folder_label in os.listdir(dir + folder_level):
            for file in os.listdir(os.path.join(dir, folder_level, folder_label)):
                name = os.path.join(dir, folder_level, folder_label, file) + ' ' + str(int(folder_label)-1) + '\n'
                train.write(name)
    train.close()

def gen_txt_origin(dir, train, test):
    folder_label_list = os.listdir(dir)
    folder_label_list.sort()
    for folder_label in folder_label_list[0:5]:
        file_list = os.listdir(os.path.join(dir, folder_label))
        file_list.sort()
        label_idx = int(folder_label[-1])-1
        for idx in range(0, 560):
            name = os.path.join(dir, folder_label, file_list[idx]) + ' ' + str(label_idx) + '\n'
            train.write(name)
        for idx in range(560, 800):
            name = os.path.join(dir, folder_label, file_list[idx]) + ' ' + str(label_idx) + '\n'
            test.write(name)
    train.close()
    test.close()


def default_loader(path):
    image = Image.open(path).convert('RGB')
    image_rgb = np.array(image)
    return image_rgb


def whitening_image(image):
    image = image.astype('float32')
    for i in range(len(image)):
        mean = np.mean(image[i, ...])
        # Use adjusted standard deviation here, in case the std == 0.
        std = np.max([np.std(image[i, ...]), 1.0/np.sqrt(42 * 42 * 3)])
        image[i, ...] = (image[i, ...] - mean) / std
    return image


def load_data(txt, transform, loader):
    fh = open(txt, 'r')
    img = []
    label = []
    for line in fh:
        line = line.strip('\n')
        words = line.split()
        img.append(loader(words[0]))
        label.append(int(words[1]))
    img = np.array(img)
    img = transform(img)
    label = np.array(label)
    return img, label



if __name__ == "__main__":
    gen_txt_flag = True
    if gen_txt_flag:
        dir = 'data/90/'
        train = open(dir+'train.txt','w')
        test = open(dir+'test.txt','w')
        gen_txt_origin(dir, train, test)
        train_data, train_label = load_data(dir+'train.txt', whitening_image, default_loader)
        test_data, test_label = load_data(dir+'test.txt', whitening_image, default_loader)





