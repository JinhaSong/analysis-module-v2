from AnalysisModule import settings
from Modules.dummy.main import Dummy
from WebAnalyzer.utils.media import frames_to_timecode

import torch
import os
from torch.autograd import Variable as V
from torchvision import transforms as trn
from torch.nn import functional as F
from PIL import Image

class Places(Dummy):
    def __init__(self):
        Dummy.__init__(self)
        code_path = os.path.dirname(os.path.abspath(__file__))
        self.model = os.path.join(code_path, 'place47.pth.tar')
        self.result = None
        self.model = torch.load(self.model)
        self.model = torch.nn.DataParallel(self.model).cuda()

        files = os.path.join(code_path, 'categories.txt')
        self.classes = list()
        with open(files) as classes_file:
            for line in classes_file:
                self.classes.append(line.strip().split(' ')[0][0:])
        self.classes = tuple(self.classes)

    def inference_by_image(self, image_path):
        result = {"place_recognition": [{"label":[]}]}
        centre_crop = trn.Compose([
            trn.CenterCrop(224),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img = Image.open(image_path)
        img = img.convert("RGB")
        place_lists = [0, 0, img.width, img.height]
        img = img.resize((256, 256), Image.ANTIALIAS)
        input_img = V(centre_crop(img).unsqueeze(0), volatile=True)

        logit = self.model.forward(input_img)
        h_x = F.softmax(logit, 1).data.squeeze()
        probs, idx = h_x.sort(0, True)

        for i in range(0, 5):
            score = probs[i]
            cls = self.classes[idx[i]]
            result["place_recognition"][0]["label"].append({
                "description": cls,
                "score": score
            })

        self.result = result

        return self.result

    def inference_by_video(self, frame_path_list, infos):
        results = []
        video_info = infos['video_info']
        frame_urls = infos['frame_urls']
        fps = video_info['fps']
        extract_fps = infos['extract_fps']

        for idx, (frame_path, frame_url) in enumerate(zip(frame_path_list, frame_urls)):
            result = self.inference_by_image(frame_path)
            result["frame_url"] = settings.MEDIA_URL + str(frame_url)[1:]
            result["frame_number"] = int((idx + 1) * fps/extract_fps)
            result["timestamp"] = frames_to_timecode((idx + 1) * fps/extract_fps, fps)

            results.append(result)

        self.result = results

        return self.result