import os
import cv2
import torch
# import torchvision.models as models
from torch import nn

import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
import numpy as np
import cv2
from PIL import Image
from Modules.places import wideresnet
from Modules.dummy.main import Dummy


class Places365(Dummy):
    path = os.path.dirname(os.path.abspath(__file__))

    def __init__(self):
        super().__init__()
        model_name = 'wideresnet18_places365.pth.tar'
        model_path = os.path.join(self.path, model_name)
        self.classes, self.labels_IO = self.load_labels()
        self.model = self.load_model(model_path)
        self.topk = 5
        self.result = None

    def recursion_change_bn(self, module):
        if isinstance(module, torch.nn.BatchNorm2d):
            module.track_running_stats = 1
        else:
            for i, (name, module1) in enumerate(module._modules.items()):
                module1 = self.recursion_change_bn(module1)
        return module


    def load_labels(self):
        file_name_category = os.path.join(self.path, 'categories_places365.txt')
        classes = list()
        with open(file_name_category) as class_file:
            for line in class_file:
                classes.append(line.strip().split(' ')[0][3:])
        classes = tuple(classes)

        # indoor and outdoor relevant
        file_name_IO = os.path.join(self.path, 'IO_places365.txt')
        with open(file_name_IO) as f:
            lines = f.readlines()
            labels_IO = []
            for line in lines:
                items = line.rstrip().split()
                labels_IO.append(int(items[-1]) - 1)  # 0 is indoor, 1 is outdoor
        labels_IO = np.array(labels_IO)

        return classes, labels_IO


    def hook_feature(self, module, input, output):
        self.features_blobs.append(np.squeeze(output.data.cpu().numpy()))

    def returnTF(self):
        tf = trn.Compose([
            trn.Resize((224, 224)),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return tf

    def load_model(self, model_path):
        self.features_blobs = []

        model = wideresnet.resnet18(num_classes=365)
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)

        # hacky way to deal with the upgraded batchnorm2D and avgpool layers...
        for i, (name, module) in enumerate(model._modules.items()):
            module = self.recursion_change_bn(model)
        model.avgpool = torch.nn.AvgPool2d(kernel_size=14, stride=1, padding=0)

        model.eval()
        # hook the feature extractor
        features_names = ['layer4', 'avgpool']  # this is the last conv layer of the resnet
        for name in features_names:
            model._modules.get(name).register_forward_hook(self.hook_feature)
        return model

    def load_image(self, image_path, tf):
        img = Image.open(image_path)
        input_img = V(tf(img).unsqueeze(0))
        return input_img

    def inference_by_image(self, image_path):
        tf = self.returnTF()

        params = list(self.model.parameters())
        weight_softmax = params[-2].data.numpy()
        weight_softmax[weight_softmax < 0] = 0

        input_img = self.load_image(image_path, tf)

        # forward pass
        logit = self.model.forward(input_img)
        h_x = F.softmax(logit, 1).data.squeeze()
        probs, idx = h_x.sort(0, True)
        probs = probs.numpy()
        idx = idx.numpy()

        result = {"frame_result": []}
        for i in range(0, self.topk):
            label = {'label':{
                'description': self.classes[idx[i]],
                'score': float(probs[i]) * 100
            }}
            result['frame_result'].append(label)

        return result
