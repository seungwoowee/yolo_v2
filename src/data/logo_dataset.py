import os
from torch.utils.data import Dataset
from src.data.data_augmentation import *
import pickle
import copy


class LOGODataset(Dataset):
    def __init__(self, img_path="../../dataset/YOLO/LOGO/images", anno_path="../../dataset/YOLO/LOGO/anno_pickle", mode="train", image_size=448, is_training=True):
        if mode in ["train", "val"]:
            self.image_path = img_path
            anno_path = os.path.join(anno_path, "LOGO_{}.pkl".format(mode))
            self.id_list_path = pickle.load(open(anno_path, "rb"))

        self.classes = ["JTBC", "KBS", "LEGO", "MARVEL", "NETFLIX", "Spiderman", "tvN", "tvn_turbo", "YouTube"]
        self.class_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.image_size = image_size
        self.num_classes = len(self.classes)
        self.num_images = len(self.id_list_path)
        self.is_training = is_training
        self.mode = mode

    def __len__(self):
        return self.num_images

    def __getitem__(self, item):
        # image_path = os.path.join(self.image_path, self.id_list_path[item]["file_name"])
        image_path = os.path.join(self.image_path, self.mode, self.id_list_path[item]["file_name"])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # cv2.imshow('tt', cv2.imread(image_path))
        # cv2.waitKey(0)
        objects = copy.deepcopy(self.id_list_path[item]["objects"])
        for idx in range(len(objects)):
            objects[idx][4] = self.class_ids.index(objects[idx][4])
        if self.is_training:
            transformations = Compose([HSVAdjust(), VerticalFlip(), Crop(), Resize(self.image_size)])
        else:
            transformations = Compose([Resize(self.image_size)])
        image, objects = transformations((image, objects))
        return np.transpose(np.array(image, dtype=np.float32), (2, 0, 1)), np.array(objects, dtype=np.float32)
