# EdgeShard

## Ubuntu
```
git clone ..
cd EdgeShard
python edge_infer_rpi5_vxxx.py
```

## Rasberry Pi
```
git clone ..
cd EdgeShard
python main_infer_ubuntu_vxxx.py
```

*** run Ubuntu Before run Rasberry Pi

V1: Single Inference Test  
Test the model with one input sentence and measure a single round of inference.  

V2: Loop-Based Inference (20 Iterations)  
Run inference in a loop for 20 iterations to evaluate consistency and performance across repeated executions.  

V3: Autoregressive Sentence Generation  
Test the model’s ability to generate continuous text by appending the predicted token and repeating inference (token-by-token generation).
