import cv2

vidcap = cv2.VideoCapture("../../../dataset/YOLO/ElClasico.mp4")
root_path = "../../../dataset/YOLO/ElClasico"
train_path = "/train/"
val_path = "/val/"
test_path = "/test/"
count = 0
train_interval = 239
val_interval = 479
test_interval = 967

while True:
    count += 1
    success, image = vidcap.read()
    if not success:
        break
    if count % train_interval == 0:
        print('Read a new frame: {}'.format(count))
        fname = "{}.jpg".format("{0:09d}".format(count))
        cv2.imwrite(root_path+train_path + fname, image)  # save frame as JPEG file
        continue
    if count % val_interval == 0:
        print('Read a new frame: {}'.format(count))
        fname = "{}.jpg".format("{0:09d}".format(count))
        cv2.imwrite(root_path+val_path + fname, image)  # save frame as JPEG file
        continue
    if count % test_interval == 0:
        print('Read a new frame: {}'.format(count))
        fname = "{}.jpg".format("{0:09d}".format(count))
        cv2.imwrite(root_path+test_path + fname, image)  # save frame as JPEG file
        continue


print("{} images are extracted in {}.".format(count, root_path))
