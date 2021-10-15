"""Details of the module.

In this file, I will check the custom reduction operator by exporing it to ONNX.

    Typical usage example:

    First create symbolic of the operator by calling -> register_custom_op()
    Then export to ONNX with  -> export_custom_op()
"""

import torch
from torch.onnx import register_custom_op_symbolic

def register_custom_op():
    """This function is to create symbolic of
    custom reduction function for exporting to ONNX

        Typical usage example:

        register_custom_op()
    """

    def my_reduction(g, layer1,layer2,layer3,layer4):
        return g.op("mydomain::Reduction", layer1, layer2, layer3, layer4)

    register_custom_op_symbolic("adagradChallenge::reduction", my_reduction, 9)


def export_custom_op():
    """This function create a sample torch module
    for the custom reduction function and uses dummy inputs
    to run the export of the operator

        Typical usage example:
        export_custom_op()
    """
    class CustomModel(torch.nn.Module):
        """This function create a sample torch module
        for the custom reduction function.

            Typical usage example:

            To create the object of this class -> CustomModel()
        """
        def forward(self, layer1, layer2, layer3, layer4):
            """this forward function calls the custom reduction operation.
            """
            return torch.ops.adagradChallenge.reduction(layer1, layer2, layer3, layer4)

    layer_1 = torch.randn(64)
    layer_2 = torch.randn(128)
    layer_3 = torch.randn(256)
    layer_4 = torch.randn(512)
    inputs = (layer_1, layer_2, layer_3, layer_4)

    onnx_save_path = '../../models/reduction.onnx'
    torch.onnx.export(CustomModel(), inputs, onnx_save_path,
                      opset_version=9,
                      example_outputs=None,
                      input_names=["X1", "X2","X3", "X4"], output_names=["Y"],
                      custom_opsets={"mydomain": 1})



torch.ops.load_library(
    "build/lib.linux-x86_64-3.6/reduction.cpython-36m-x86_64-linux-gnu.so")

register_custom_op()
export_custom_op()
