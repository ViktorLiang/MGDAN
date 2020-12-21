import os

import argparse
import cv2
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models

from models.sharedTask.mask_edge_patch import ResNet, Bottleneck


class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers, with_edge=True):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)
        self.with_edge = with_edge

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x, exec_list=None, feature_model_name="layer4"):
        target_activations = []
        if exec_list is None:
            for name, module in self.model._modules.items():
                if module == self.feature_module:
                    target_activations, x = self.feature_extractor(x)
                elif "avgpool" in name.lower():
                    x = module(x)
                    x = x.view(x.size(0),-1)
              
                else:
                    x = module(x)
             
        else:
            layers_name = ["layer1", "layer2", "layer3", "layer4", "layer5"]
            assert feature_model_name in layers_name

            edge_layer_name = "edge_layer"
            decoder_layer_name = "layer6"
            fuse_layer_name = "layer7"
            rescore_layer = "seg_rescore_conv"

            layers_features = []
            print("feature_model_name:", feature_model_name)
            for i, name in enumerate(exec_list):
                print(i, name)
                module = self.model._modules[name]
                if name in exec_list[:10]: # before maxpool
                    x = module(x)

                if name == feature_model_name:
                    target_activations, x = self.feature_extractor(x)
                    if name in layers_name:
                        layers_features.append(x)
                        continue

                if name in layers_name:
                    x = module(x)
                    layers_features.append(x)

                if name == edge_layer_name:
                    preds_from_edge, x_fg, x_edges = module(layers_features[0], layers_features[1], layers_features[2], layers_features[4])
                    if self.with_edge:
                        edge_pred, fg_pred, seg_from_edge = preds_from_edge
                    else:
                        fg_pred, seg_from_edge = preds_from_edge


                if name == decoder_layer_name:
                    seg_from_dec, x_decod = module(layers_features[4], layers_features[0], xfg=x_fg)

                if name == fuse_layer_name:
                    x_cat = torch.cat([x_decod, x_edges], dim=1)
                    seg_from_fuse = module(x_cat)

                if name == rescore_layer:
                    if self.with_edge:
                        seg_cat = torch.cat([seg_from_dec, seg_from_fuse, seg_from_edge], dim=1)
                    else:
                        seg_cat = torch.cat([seg_from_dec, seg_from_edge], dim=1)

                    seg_from_rescore = module(seg_cat)
            # return x
                
        
        # return target_activations, x
        if self.with_edge:
            return target_activations, seg_from_fuse
        else:
            return target_activations, x_decod


def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(True)
    return input


def show_cam_on_image(img, mask, imname):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite(imname+"_cam.jpg", np.uint8(255 * cam))


class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda, with_edge=True):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names, with_edge=with_edge)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None, exec_list=[], feature_model_name="layer4"):
        if self.cuda:
            features, output = self.extractor(input.cuda(), exec_list=exec_list, feature_model_name=feature_model_name)
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())
        
        # output torch.Size([1, 20, 56, 56])
        # one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)

        one_hot = np.zeros((1, output.size()[1], output.size()[2], output.size()[3]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, input.shape[2:])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


class GuidedBackpropReLU(Function):

    @staticmethod
    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input),
                                   torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output,
                                                 positive_mask_1), positive_mask_2)

        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        def recursive_relu_apply(module_top):
            for idx, module in module_top._modules.items():
                recursive_relu_apply(module)
                if module.__class__.__name__ == 'ReLU':
                    module_top._modules[idx] = GuidedBackpropReLU.apply
                
        # replace ReLU with GuidedBackpropReLU
        recursive_relu_apply(self.model)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            output = self.forward(input.cuda())
        else:
            output = self.forward(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())
        seg_preds = output[0]
        edge_preds = output[1]
        fg_preds = output[2]
        seg_output = seg_preds[1]

        one_hot = np.zeros((1, seg_output.size()[1], seg_output.size()[2], seg_output.size()[3]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * seg_output)
        else:
            one_hot = torch.sum(one_hot * seg_output)

        # self.model.features.zero_grad()
        # self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        output = input.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        return output



def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img*255)

def get_model(load_pth, with_edge=True, with_dcn=True, grad_layer=13):
    # def Res_Deeplab(num_classes=7, with_dcn=[1,1,1], with_rescore=True, rescore_k_size=31):
    # model = Res_Deeplab(num_classes=20, with_dcn=[1,1,1], with_rescore=True, rescore_k_size=31)
    dcn_list = [1,1,1] if with_dcn else [0,0,0]
    with_mask_edge = with_edge

    model = ResNet(Bottleneck, [3,4,23,3], 20,
                    with_dcn=dcn_list, 
                    with_edge=with_edge, 
                    with_mask_atten=True,
                    with_mask_edge=with_mask_edge,
                    with_mask_pars=True,
                    with_rescore=True,
                    rescore_k_size=31)

    state_dicts = torch.load(load_pth)
    model_stat = model.state_dict()
    for k, v in state_dicts.items():
        ks = k.split(".")
        model_stat[".".join(ks[1:])] = state_dicts[k] 
        
    model.load_state_dict(model_stat)

    #complete model
    # exec_list = ["conv1", "bn1", "relu1", "conv2", "bn2", "relu2", "conv3", "bn3", "relu3", "maxpool",#10
    #                 "layer1", "layer2", "layer3", "layer4", "layer5", #15
    #                 "edge_layer", "layer6", "layer7", "seg_rescore_conv"] #19

    #no edge
    if not with_edge:
        exec_list = ["conv1", "bn1", "relu1", "conv2", "bn2", "relu2", "conv3", "bn3", "relu3", "maxpool",#10
                    "layer1", "layer2", "layer3", "layer4", "layer5", #15
                    "edge_layer","layer6", "seg_rescore_conv"] #17
    else:
        exec_list = ["conv1", "bn1", "relu1", "conv2", "bn2", "relu2", "conv3", "bn3", "relu3", "maxpool",#10
                    "layer1", "layer2", "layer3", "layer4", "layer5", #15
                    "edge_layer", "layer6", "layer7", "seg_rescore_conv"] #19
    
    return model, exec_list, exec_list[grad_layer]
    # grad_cam = GradCam(model=model, feature_module=model.layer4, \
                    #    target_layer_names=["2"], use_cuda=False)


def get_args():
  
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, 
            default='',
            help='Input image path')
    
    parser.add_argument('--param-path', type=str, default="", help='Input image path')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args


if __name__ == '__main__':
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """
    project = '/home/ly/workspace/git/segmentation/human_parsing/TextureShapeParsing_CIHP/'
    img_root = '/home/ly/data/datasets/human_parsing/CIHP/CIHP/instance-level_human_parsing/Validation/Images/'

    params_full_patch = project+'/snapshots/shared_test/test01_CIHP_grids_448/no_edgeParsRefine/bestiou_CIHP_epoch_102.pth'
    params_full = project+'/snapshots/shared_test/test01_CIHP/CIHP_epoch_199.pth'
    params_noedge = project+'/snapshots/shared_test/test01_CIHP_grids_448_noedge/CIHP_epoch_101.pth'
    params_noedge_nodcn = project+'/snapshots/shared_test/test01_CIHP_grids_448_noedge_nodcn/CIHP_epoch_97.pth'
    params_dict = {
        "full_patch_model":params_full_patch,
        "full_model":params_full,
        "noedge_model":params_noedge,
        "noedge_nodcn_model":params_noedge_nodcn,
    }
    

    args = get_args()

    # Can work with any model, but it assumes that the model has a
    # feature method, and a classifier method,
    # as in the VGG models in torchvision.

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested index.
    # target_index = 15 #left arm
    # target_index = 6 #dress
    # target_index = 12 #skirt
    # target_index = 1 #hat
    # target_index = 9 #pants
    # target_index = 18 #right-shoe
    # target_index = 17 #right-leg
    # target_index = 16 #left-leg
    # target_index = 5 #upper clothes
    # target_index = 9 #coat
    # target_index = 7
    target_index = 8 #socks

    with_edge = False
    with_dcn = True
    model_name = 'noedge_model'
    args.param_path = params_dict[model_name]
    args.image_path = img_root+"/"+"0009669"+".jpg"
    print(args.param_path)
    print(args.image_path)

    imname = os.path.basename(args.image_path)
    imname = imname.split(".")[0]+"_"+model_name+"_label"+str(target_index)
    save_path = "./"+model_name+"/"+imname
    print("save to:",save_path)

    model, exec_list, feature_model_name = get_model(args.param_path, with_edge=with_edge, with_dcn=with_dcn, grad_layer=13)
    print("grad model name:", feature_model_name)
    model.cuda()
    grad_model = model.layer4

    grad_cam = GradCam(model=model, feature_module=grad_model, \
                       target_layer_names=["2"], use_cuda=args.use_cuda, with_edge=with_edge)

    img = cv2.imread(args.image_path, 1)
    img = np.float32(cv2.resize(img, (448, 448))) / 255
    input = preprocess_image(img)
  


    mask = grad_cam(input, target_index, exec_list=exec_list, feature_model_name=feature_model_name)

    show_cam_on_image(img, mask, save_path)

    #gradients respect to input image 
    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
    gb = gb_model(input, index=target_index)
    gb = gb.transpose((1, 2, 0))
    cam_mask = cv2.merge([mask, mask, mask])
    cam_gb = deprocess_image(cam_mask*gb)
    gb = deprocess_image(gb)

    cv2.imwrite(save_path+'_gb.jpg', gb)
    cv2.imwrite(save_path+'_cam_gb.jpg', cam_gb)