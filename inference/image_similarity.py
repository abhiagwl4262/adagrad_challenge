""" Module to check similarity of images stroed in a folder.
I will use a modified resnet to get final [1,512]
size feature map for each image. I then use cosine similarity function
from torch to get the final matching score.

Then based on the threshold 0.75, I will categorise if the images matched or not.
Running this will write the final matching matrix, which has result of matching
between ith image to jth image
"""
import time
import os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from resnet import resnet18

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

torch.ops.load_library(
    "pytorch_custom_op/build/lib.linux-x86_64-3.6/reduction.cpython-36m-x86_64-linux-gnu.so")

from torch.onnx import register_custom_op_symbolic

def register_custom_op():
    """This function is to create symbolic of
    custom reduction function for exporting to ONNX

        Typical usage example:

        register_custom_op()
    """

    def my_reduction(g, layer1,layer2,layer3,layer4):
        return g.op("mydomain::Reduction", layer1, layer2, layer3, layer4)

    register_custom_op_symbolic("adagradChallenge::reduction", my_reduction, 11)

def reduction(features):
    """Implement Reduction with torch.repeat_interleave
    function
    """
    begin = time.time()
    feat1 = torch.repeat_interleave(features[0], torch.Tensor([8]).long())
    feat2 = torch.repeat_interleave(features[1], torch.Tensor([4]).long())
    feat3 = torch.repeat_interleave(features[2], torch.Tensor([2]).long())
    end = time.time()
    print(f"Time taken by python interleaves is {end - begin}")
    
    begin = time.time()    
    out = (feat1+feat2+feat3+features[-1])/4.0
    end = time.time()
    print(f"Time taken by python sum mean is {end - begin}")

    return out
    # return torch.cat((feat1, feat2, feat3, features[-1].cpu()), dim=0).mean(0)

class ReductionResnet(torch.nn.Module):
    """this is implementation of reductionresnet.
    A modified class from resnet18.
    """
    def __init__(self, backbone):
        super(ReductionResnet, self).__init__()
        self.backbone = backbone

    def forward(self, inp):
        """forward function of the ReductionResnet.
        which take image as input and return [1, 512] size torch tensor
        """
        features = self.backbone(inp)    
        # avgpool1 = F.adaptive_avg_pool2d(features[0], (1, 1)).squeeze(-1).squeeze(-1).squeeze(0)
        # avgpool2 = F.adaptive_avg_pool2d(features[1], (1, 1)).squeeze(-1).squeeze(-1).squeeze(0)
        # avgpool3 = F.adaptive_avg_pool2d(features[2], (1, 1)).squeeze(-1).squeeze(-1).squeeze(0)
        # avgpool4 = F.adaptive_avg_pool2d(features[3], (1, 1)).squeeze(-1).squeeze(-1).squeeze(0)

        avgpool1 = F.adaptive_avg_pool2d(features[0], (1, 1)).view(-1,)
        avgpool2 = F.adaptive_avg_pool2d(features[1], (1, 1)).view(-1,)
        avgpool3 = F.adaptive_avg_pool2d(features[2], (1, 1)).view(-1,)
        avgpool4 = F.adaptive_avg_pool2d(features[3], (1, 1)).view(-1,)

        #begin = time.time()
        output = torch.ops.adagradChallenge.reduction(avgpool1.to("cpu"),avgpool2.to("cpu"),avgpool3.to("cpu"),avgpool4.to("cpu"))
        #end = time.time()
        #print("time taken to process c++ reduction is ", (end-begin))

        return output.unsqueeze(0)

def run(args):
    """The function which runs the full flow.
    loads data, create model object,
    returns concated output of all the images
    """
    backbone = resnet18(pretrained=True)
    model = ReductionResnet(backbone).to(device)
    model.eval()
    outputs = []
    data_dir = "../Challenge_images/"
    img_paths = os.listdir(data_dir)
    img_paths.sort(key = lambda x:x.split(".")[0].zfill(2))
    img_paths = [os.path.join(data_dir,path) for path in img_paths]
    for img_path in img_paths:
        try :
            img = transforms.ToTensor()(transforms.Resize((416,416))(Image.open(img_path))).unsqueeze(0)
        except:
            print("There is some issue with ", img_path)
            continue
        img = img.to(device)
        output = model(img)
        if args.onnx:
            torch.onnx.export(model, img, "../models/model.onnx", opset_version=11)
            args.onnx=False
        if not len(outputs):
            outputs.append(output)
        else:
            outputs[0] = torch.cat((outputs[0], output), dim=0)
        
    return outputs[0]

def get_scores(tensors):
    """This function takes the output of run function
    and calculate cosine similarity scores between all the images.
    score -> (n x n) where n = num_images
    from scores I classify if ot is a match or not
    based on threshold
    """
    num_tensors = tensors.shape[0]
    scores = np.zeros((num_tensors, num_tensors))
    matches = np.zeros((num_tensors, num_tensors))
    for i in range(num_tensors):
        scores[i,:] = F.cosine_similarity(tensors[i].unsqueeze(0).repeat(num_tensors, 1), tensors).detach().numpy()
    # for i in range(num_tensors):
    #     for j in range(num_tensors):
    #         scores[i][j] = F.cosine_similarity(tensors[i].unsqueeze(0), tensors[j].unsqueeze(0)).detach().numpy()
    print(scores)
    matches[np.where(scores>0.75)] = 1
    np.save("matches.npy", matches)

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser(description="My parser")
    parser.add_argument("--onnx",type=bool)
    args = parser.parse_args()
    register_custom_op()
    tensors = run(args)
    get_scores(tensors)
