import os
import numpy as np
import cv2

class DataLoader:
    def __init__(self, folder_path, batch_size):
        self.folder_path = folder_path
        self.batch_size = batch_size

        self.image_paths = []
        self.labels = []

        for file in os.listdir(self.folder_path):
            img_path = os.path.join(self.folder_path, file)
            self.image_paths.append(img_path)

        self.image_paths = np.array(self.image_paths)
        self.n_samples = len(self.image_paths)
        self.indices = np.arange(self.n_samples)

    def __len__(self):
        return int(np.ceil(self.n_samples / self.batch_size))
    
    def __iter__(self):
        for i in range(0, self.n_samples, self.batch_size):
            batch_idx = self.indices[i:i + self.batch_size]

            images = []
            
            for j in batch_idx:
                img = cv2.imread(self.image_paths[j])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)

            yield np.array(images)