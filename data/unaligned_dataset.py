""" import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import random


class UnalignedDataset(BaseDataset):
    
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    

    def __init__(self, opt):
        Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + "A")  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + "B")  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))  # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))  # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == "BtoA"
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc  # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc  # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))
    def myopen(self,path):
        "Open image and return none if image corrupted"
        try:
            img = Image.open(path)
            return img.convert("RGB")
        except Exception as e:
            print(f"Error opening image {path}: {e}")
            return None 
               
    def __getitem__(self, index):
        Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:  # make sure index is within then range
            index_B = index % self.B_size
        else:  # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = self.myopen(A_path)
        B_img = self.myopen(B_path)
        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        return {"A": A, "B": B, "A_paths": A_path, "B_paths": B_path}

    def __len__(self):
        Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        
        return max(self.A_size, self.B_size) """
    



import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # tolerate truncated JPEG/PNG
import random

class UnalignedDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + "A")
        self.dir_B = os.path.join(opt.dataroot, opt.phase + "B")

        # 1) Build file lists (as before)
        A_all = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        B_all = sorted(make_dataset(self.dir_B, opt.max_dataset_size))

        # 2) Filter by extensions (avoid non-image files)
        valid_exts = {".jpg", ".jpeg", ".png"}
        A_all = [p for p in A_all if os.path.splitext(p)[1].lower() in valid_exts]
        B_all = [p for p in B_all if os.path.splitext(p)[1].lower() in valid_exts]

        # 3) Verify headers once; exclude corrupted/unreadable images
        def _is_valid_image(path):
            try:
                with Image.open(path) as img:
                    img.verify()  # fast header check
                return True
            except Exception as e:
                # Log the first time only; avoid flooding
                print(f"[Data] Excluding bad image: {path} ({e})")
                return False

        self.A_paths = [p for p in A_all if _is_valid_image(p)]
        self.B_paths = [p for p in B_all if _is_valid_image(p)]

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

        print(f"[Data] {opt.phase}: A={self.A_size}/{len(A_all)} kept, "
              f"B={self.B_size}/{len(B_all)} kept")

        # Transforms
        btoA = self.opt.direction == "BtoA"
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

    def _safe_open_rgb(self, path):
        try:
            img = Image.open(path)
            return img.convert("RGB")
        except Exception as e:
            print(f"[Data] Skipping bad image at runtime: {path} ({e})")
            return None

    def __getitem__(self, index):
        # Retry both A and B if either fails, to avoid bias
        max_retries = 50
        a_idx = index % self.A_size
        b_idx = (index % self.B_size) if self.opt.serial_batches else random.randint(0, self.B_size - 1)

        for _ in range(max_retries):
            A_path = self.A_paths[a_idx]
            B_path = self.B_paths[b_idx]

            A_img = self._safe_open_rgb(A_path)
            B_img = self._safe_open_rgb(B_path)

            if A_img is not None and B_img is not None:
                A = self.transform_A(A_img)
                B = self.transform_B(B_img)
                return {"A": A, "B": B, "A_paths": A_path, "B_paths": B_path}

            # Change BOTH sides on failure
            if self.opt.serial_batches:
                a_idx = (a_idx + 1) % self.A_size
                b_idx = (b_idx + 1) % self.B_size
            else:
                a_idx = random.randint(0, self.A_size - 1)
                b_idx = random.randint(0, self.B_size - 1)

        raise RuntimeError("[Data] Could not fetch a valid A/B pair after retries. Consider cleaning data.")

    def __len__(self):
        return max(self.A_size, self.B_size)

