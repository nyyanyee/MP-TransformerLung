import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models
from torchvision import transforms
from utils import GradCAM, show_cam_on_image, center_crop_img

os.environ["CUDA_VISIBLE_DEVICES"] = '4'


def main():
    model = torch.load("/icislab/volume1/FWen/my_model/best_model.pth")
    #model = model.module

    target_layers = [model.layer4[2].conv3]

    # model = models.vgg16(pretrained=True)
    # target_layers = [model.features]

    # model = models.resnet34(pretrained=True)
    # target_layers = [model.layer4]

    # model = models.regnet_y_800mf(pretrained=True)
    # target_layers = [model.trunk_output]

    # model = models.efficientnet_b0(pretrained=True)
    # target_layers = [model.features]

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.6619453, 0.4676412, 0.6778352], std=[0.1554726, 0.22549164, 0.14226793])])
    # set input and output directories
    input_dir = "/icislab/volume1/ubeihang/pytorch-3.7/code/luad/heatmap"
    output_dir = "/icislab/volume1/ubeihang/pytorch-3.7/code/luad/res101-heatmap"
    # loop over subdirectories and images
    for subdir in os.listdir(input_dir):
        sub_input_dir = os.path.join(input_dir, subdir)
        sub_output_dir = os.path.join(output_dir, subdir)
        if not os.path.exists(sub_output_dir):
            os.makedirs(sub_output_dir)
        for filename in os.listdir(sub_input_dir):
            # load image
            img_path = os.path.join(sub_input_dir, filename)
            assert os.path.exists(img_path), "file: '{}' does not exist.".format(img_path)
            img = Image.open(img_path).convert('RGB')
            img = np.array(img, dtype=np.uint8)
            # img = center_crop_img(img, 224)

            # [C, H, W]
            img_tensor = data_transform(img)
            # expand batch dimension
            # [C, H, W] -> [N, C, H, W]
            input_tensor = torch.unsqueeze(img_tensor, dim=0)

            cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
            # target_category = 281  # tabby, tabby cat
            # target_category = 254  # pug, pug-dog

            grayscale_cam = cam(input_tensor=input_tensor)

            grayscale_cam = grayscale_cam[0, :]
            visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                              grayscale_cam,
                                              use_rgb=True)

            # save output image
            output_path = os.path.join(sub_output_dir, filename)
            plt.imsave(output_path, visualization)


if __name__ == '__main__':
    main()
