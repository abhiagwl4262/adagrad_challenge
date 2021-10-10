""" Module to check similarity of images stroed in a folder.
I will use a modified resnet to get final [1,512]
size feature map for each image. I then use cosine similarity function
from torch to get the final matching score.

Then based on the threshold 0.75, I will categorise if the images matched or not.
Running this will write the final matching matrix, which has result of matching
between ith image to jth image
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from resnet import resnet18

torch.ops.load_library(
    "pytorch_custom_op/build/lib.linux-x86_64-3.6/reduction.cpython-36m-x86_64-linux-gnu.so")

def reduction(features):
    """Implement Reduction with torch.repeat_interleave
    function
    """
    feat1 = torch.repeat_interleave(features[0].cpu(), torch.Tensor([8]).long())
    feat2 = torch.repeat_interleave(features[1].cpu(), torch.Tensor([4]).long())
    feat3 = torch.repeat_interleave(features[2].cpu(), torch.Tensor([2]).long())
    return (feat1+feat2+feat3+features[-1].cpu())/4.0

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
        avgpool1 = F.adaptive_avg_pool2d(features[0], (1, 1)).squeeze(-1).squeeze(-1).squeeze(0)
        avgpool2 = F.adaptive_avg_pool2d(features[1], (1, 1)).squeeze(-1).squeeze(-1).squeeze(0)
        avgpool3 = F.adaptive_avg_pool2d(features[2], (1, 1)).squeeze(-1).squeeze(-1).squeeze(0)
        avgpool4 = F.adaptive_avg_pool2d(features[3], (1, 1)).squeeze(-1).squeeze(-1).squeeze(0)
        output = torch.ops.adagradChallenge.reduction(avgpool1.to("cpu"),avgpool2.to("cpu"),avgpool3.to("cpu"),avgpool4.to("cpu"))
        # output2 = reduction((avgpool1,avgpool2,avgpool3,avgpool4))

        return output.unsqueeze(0)

def run():
    """The function which runs the full flow.
    loads data, create model object,
    returns concated output of all the images
    """
    backbone = resnet18(pretrained=True)
    model = ReductionResnet(backbone).to("cuda:0")
    model.eval()
    outputs = []
    data_dir = "../Challenge_images/train/0/"
    img_paths = os.listdir(data_dir)
    img_paths.sort(key = lambda x:x.split(".")[0].zfill(2))
    img_paths = [os.path.join(data_dir,path) for path in img_paths]
    for img_path in img_paths:
        try :
            img = transforms.ToTensor()(transforms.Resize((416,416))(Image.open(img_path))).unsqueeze(0)
        except:
            print("There is some issue with ", img_path)
            continue
        img = img.to("cuda:0")
        output = model(img)
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

tensors = run()
get_scores(tensors)
