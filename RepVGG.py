import cv2
import torch

from torchvision import transforms
from repvgg_attention import create_RepVGG_A0
import json


# create_RepVGG_A0模型
def model_load(weights_path, num_classes, device, att_type, use_se):
    model = create_RepVGG_A0(num_classes=num_classes, att_type=att_type, use_se=use_se).to(device)
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    model.load_state_dict(torch.load(weights_path, map_location=device))
    return model


class RepVGG(object):
    # 初始化操作，加载模型
    def __init__(self, num_classes=None, weights_path='weights/ShuffleAttention-Oct22_15-54-53-best.pth',
                 att_type='ShuffleAttention', use_se=True, json_path='./class_indices.json', device="cuda:0",
                 **kwargs):
        self.weights_path = weights_path
        self.num_classes = num_classes
        self.device = device
        self.att_type = att_type
        self.use_se = use_se
        self.model = model_load(self.weights_path, self.num_classes, self.device, self.att_type, self.use_se).eval()
        self.data_transform = transforms.Compose([transforms.Resize(256),
                                                  transforms.CenterCrop(224),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.class_indict = json.load(open(json_path, "r"))

    # 推理部分
    def infer(self, inImg):
        img = self.data_transform(inImg)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)
        with torch.no_grad():
            output = torch.squeeze(self.model(img.to(self.device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()
        # print_res = "class: {}   prob: {:.3}".format(self.class_indict[str(predict_cla)],
        #                                              predict[predict_cla].numpy())
        # print(print_res)
        return predict_cla, predict[predict_cla].numpy()

    # 画图部分
    def plot_one_img(self, img, cla=0, prob=0):
        label_cla = "class:{}".format(self.class_indict[str(cla)])
        label_prob = "prob:{:.3}".format(prob)
        # img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        cv2.putText(img, label_cla, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (35, 36, 117), 2)
        cv2.putText(img, label_prob, (8, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (35, 36, 117), 2)
        return img
