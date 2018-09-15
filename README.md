# DeepRec
In this repository, we implement many recent deep learning based recommendation models with Tensorflow.


# Collaborators
[Shuai Zhang](https://sites.google.com/view/shuaizhang/home), [Yi Tay](https://sites.google.com/view/yitay/home), [Bin Wu](https://github.com/wubin7019088)


# Implemented Algorithms
We implemented both rating estimation and top-n recommendation models.
* I-AutoRec and U-AutoRec (www'15)
* CDAE (WSDM'16)
* NeuMF (WWW'17)
* CML (WWW'17)
* LRML (WWW'18) (DRAFT ONLY, testing will come soon)
* NFM (SIGIR'17)
* NNMF (arxiv)
* etc.

You can run this code from Test/testItemRanking.py or Test/testRatingPred.py

# Requirements
* Tensorflow 1.7+, Python 3.5+, numpy, scipy, sklearn, pandas

# To do
* Add more models
* Different Evaluation Protocals
* Code Refactor

# Citation

To acknowledge use of this open source package in publications, please cite the
following paper:

```
@article{zhang2017deeprec,
  title={Deep learning based recommender system: A survey and new perspectives},
  author={Zhang, Shuai and Yao, Lina and Sun, Aixin and Tay, Yi},
  journal={arXiv preprint arXiv:1707.07435},
  year={2017}
}
```
Thank you for your support!


Contributions and issues are always welcome. You can also contact me via email: cheungshuai@outlook.com
