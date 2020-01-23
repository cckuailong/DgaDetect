# DgaDetect

DGA(Domain Generate Algorithm) Detect via Deeplearning Tech.

- Train Accuracy:   99.41%
- Test Accuracy:    98%

## Structure

- data/

To store data

- dga_family/

Different kinds of DGA generators

- handle_data.py

Use dga_family and data/top-1m.csv to make the train dataset(traindata.pkl)

- tflearn/

Use tflearn to train via LSTM and AMSGrad

![tflearn](https://github.com/cckuailong/DgaDetect/blob/master/tflearn/result.png)

- keras

Use keras to train via LSTM and NAdam

![keras](https://github.com/cckuailong/DgaDetect/blob/master/keras/result.png)
