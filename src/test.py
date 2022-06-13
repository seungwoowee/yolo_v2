import glob
import argparse
import pickle
import cv2
from src.util.utils import *
from src.models.yolo_net import Yolo

CLASSES = ["JTBC", "KBS", "LEGO", "MARVEL", "NETFLIX", "Spiderman", "tvN", "tvn_turbo", "YouTube"]


def get_args():
    parser = argparse.ArgumentParser("You Only Look Once: Unified, Real-Time Object Detection")
    parser.add_argument("--image_size", type=int, default=448, help="The common width and height for all images")
    parser.add_argument("--conf_threshold", type=float, default=0.35)  #
    parser.add_argument("--nms_threshold", type=float,
                        default=0.5)  # Non-maximum Suppression: remove objects considered identical
    parser.add_argument("--pre_trained_model_type", type=str, choices=["model", "params"], default="params")
    parser.add_argument("--pre_trained_model_path", type=str,
                        default="../experiments/yolo_v2_448/trained_models/yolo_v2_448_10.pth")
    # parser.add_argument("--input", type=str, default="../../dataset/YOLO/input_images_with_LOGO/images/test")
    parser.add_argument("--input", type=str, default="../../dataset/YOLO/ElClasico/test")
    parser.add_argument("--output", type=str, default="../test_out")
    parser.add_argument("--num_class", type=int, default=9)

    args = parser.parse_args()
    return args


def test(opt):
    if torch.cuda.is_available():
        if opt.pre_trained_model_type == "model":
            model = torch.load(opt.pre_trained_model_path)
        else:
            model = Yolo(opt.num_class)
            load_net = torch.load(opt.pre_trained_model_path)

            load_net_clean = OrderedDict()
            for k, v in load_net.items():
                if k.startswith('module.'):
                    load_net_clean[k[7:]] = v
                else:
                    load_net_clean[k] = v
            model.load_state_dict(load_net_clean, strict=True)
    else:
        if opt.pre_trained_model_type == "model":
            model = torch.load(opt.pre_trained_model_path, map_location=lambda storage, loc: storage)
        else:
            model = Yolo(opt.num_class)
            load_net = torch.load(opt.pre_trained_model_path)

            load_net_clean = OrderedDict()
            for k, v in load_net.items():
                if k.startswith('module.'):
                    load_net_clean[k[7:]] = v
                else:
                    load_net_clean[k] = v
            model.load_state_dict(load_net_clean, strict=True)
    model.eval()
    colors = pickle.load(open("pallete", "rb"))
    # cnt = 0
    for image_path in glob.iglob(opt.input + os.sep + '*.*'):
        # # swee
        # cnt += 1
        # if cnt != 61:
        #     continue
        ##
        if "prediction" in image_path:
            continue
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        image = cv2.resize(image, (opt.image_size, opt.image_size))
        image = np.transpose(np.array(image, dtype=np.float32), (2, 0, 1))
        image = image[None, :, :, :]
        width_ratio = float(opt.image_size) / width
        height_ratio = float(opt.image_size) / height
        data = Variable(torch.FloatTensor(image))
        if torch.cuda.is_available():
            data = data.cuda()
        with torch.no_grad():
            logits = model(data)
            predictions = post_processing(logits, opt.image_size, CLASSES, model.anchors, opt.conf_threshold,
                                          opt.nms_threshold)
        if len(predictions) != 0:
            predictions = predictions[0]
            output_image = cv2.imread(image_path)
            preds = ''
            for pred in predictions:
                xmin = int(max(pred[0] / width_ratio, 0))
                ymin = int(max(pred[1] / height_ratio, 0))
                xmax = int(min((pred[0] + pred[2]) / width_ratio, width))
                ymax = int(min((pred[1] + pred[3]) / height_ratio, height))
                color = colors[CLASSES.index(pred[5])]
                cv2.rectangle(output_image, (xmin, ymin), (xmax, ymax), color, 2)
                text_size = cv2.getTextSize(pred[5] + ' : %.2f' % pred[4], cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
                cv2.rectangle(output_image, (xmin, ymin), (xmin + text_size[0] + 3, ymin + text_size[1] + 4), color, -1)
                cv2.putText(
                    output_image, pred[5] + ' : %.2f' % pred[4],
                    (xmin, ymin + text_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1,
                    (255, 255, 255), 1)
                # print("Object: {}, Bounding box: ({},{}) ({},{})".format(pred[5], xmin, xmax, ymin, ymax))
                for p in pred[5:]:  # 예측된 class 기록
                    preds = preds + p + '_'

            cv2.imwrite(opt.output + '\\' + preds + image_path[len(opt.input) + 1:], output_image)
        else:
            output_image = cv2.imread(image_path)
            cv2.imwrite(opt.output + '/failed' + image_path[len(opt.input):], output_image)


if __name__ == "__main__":
    opt = get_args()
    test(opt)
