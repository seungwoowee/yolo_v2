import time
import os
import cv2
import numpy as np
import random
import pickle

print("Process Start.")

start_time = time.time()

# input image list
input_dir = 'input_images2'
input_folders = os.listdir(input_dir)

# logo list
logo_dir = 'logo_images'
logo_files = os.listdir(logo_dir)

# output directory
out_dir = "data/LOGO/images2"
# if out_dir not in os.listdir():
#     os.mkdir(out_dir)

# set image/logo ratio factor
logo_ratio_factor = 10

# set logo dimming factor
logo_dimming_factors = np.arange(1, 2, 0.25)  # 1: origin

for input_folder in input_folders:
    id_list_path = []
    logo_idx = 0
    for logo_file in logo_files:
        logo_idx = logo_idx + 1
        # logo read
        logo = cv2.imread(logo_dir + '/' + logo_file)
        logo_y, logo_x = logo.shape[0], logo.shape[1]

        for input_file in os.listdir(os.path.join(input_dir, input_folder)):
            # check whether image files            
            exp = input_file.strip().split('.')[-1]
            if exp not in "JPG jpg JPEG jpeg PNG png BMP bmp":
                continue

            # image read
            image = cv2.imread(os.path.join(input_dir, input_folder) + "/" + input_file)

            # get image size
            image_y, image_x = image.shape[0], image.shape[1]

            # set logo size        
            if logo_x / image_x > logo_y / image_y:
                new_logo_x = int(image_x / logo_ratio_factor)
                # new_logo_y : logo_y = new_logo_x : logo_x
                new_logo_y = int(logo_y * (new_logo_x / logo_x))
            else:
                new_logo_y = int(image_y / logo_ratio_factor)
                new_logo_x = int(logo_x * (new_logo_y / logo_y))

            # resize logo image
            for logo_dimming_factor in logo_dimming_factors:
                resized_logo = (cv2.resize(logo, (new_logo_x, new_logo_y))[:] // logo_dimming_factor).astype(np.uint8)

                # set logo position
                x_min = int(image_x / 50) + random.randint(0, image_x - new_logo_x - int(image_x / 50))
                y_min = int(image_y / 50) + random.randint(0, image_y - new_logo_y - int(image_y / 50))
                x_max = x_min + new_logo_x
                y_max = y_min + new_logo_y

                logo_inserted = np.empty((image_y, image_x, 3), np.uint8)
                logo_inserted[y_min:y_max, x_min:x_max] = resized_logo
                # cv2.imshow('tt', logo_inserted)
                # cv2.waitKey(0)
                out_image = image.astype(np.uint16) + logo_inserted.astype(np.uint16)
                out_image[out_image > 255] = 255
                out_image = out_image.astype(np.uint8)

                # image save
                cv2.imwrite(os.path.join(out_dir, input_folder, logo_file.split('.')[0] + '_dimming_{}_'.format(
                    logo_dimming_factor) + input_file), out_image)
                id_list_path.append(
                    {'file_name': logo_file.split('.')[0] + '_dimming_{}_'.format(logo_dimming_factor) + input_file,
                     'objects': [[x_min, y_min, x_max, y_max, logo_idx]]})

    with open('data/LOGO/anno_pickle2/LOGO_' + input_folder + '.pkl', 'wb') as f:
        pickle.dump(id_list_path, f, pickle.HIGHEST_PROTOCOL)

end_time = time.time()
print("Process Done. " + str(end_time - start_time) + " seconds.")
