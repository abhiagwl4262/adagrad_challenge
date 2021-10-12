# adagrad_challenge
### To run the python code on ubuntu
cd research
python3 image_similarity.py (if GPU is available, this whole code will run on GPU.                         
                             The score should be 51*51 but 8.jpg is not valid
                             so, it will print 50*50 score matrix.
                             matches.npy has the final matching decision based on threshold.
                             The images were load in sorted manner. )


## To run the python code on ubuntu

#### 1. To build the cpp code
cd inference/pytorch_custom_op/
python3 setup.py install

#### 2. Exporting the reduction module after build is successful
python3 export_reduction.py (this will write reduction.onnx into ../models/)

#### 3. To run the code with custom redcution operation
cd inference/
python3 image_similarity.py --onnx True (This will write full model ONNX file into ../models/)
                                        (if GPU is available, the backbone along with 
                                        adaptive average pooling will run on GPU.
                                        The reduction operation is running on CPU.                           
                                        The score should be 51*51 but 8.jpg is not valid
                                        so, it will print 50*50 score matrix.
                                        matches.npy has the final matching decision based on threshold.
                                        The images were load in sorted manner. )
