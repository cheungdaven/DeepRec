# DeepRec
In this repository, we implement many recent deep learning based recommendation models with Tensorflow.

# Implemented Algorithms
We implemented both rating estimation and top-n recommendation models.
* I-AutoRec and U-AutoRec (www'15)
* CDAE (WSDM'16)
* NeuMF (WWW'17)
* CML (WWW'17)
* LRML (WWW'18) (DRAFT ONLY, testing will come soon)
* NFM (SIGIR'17)
* NNMF (arxiv)
* PRME (IJCAI 2015)
* CASER (WSDM 2018)
* AttRec (AAAI 2019 RecNLP)
and so on.

You can run this code from Test/test_item_ranking.py, Test/test_rating_pred.py, or Test/testSeqRec.py

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
@article{zhang2019deeprec,
  title={Deep learning based recommender system: A survey and new perspectives},
  author={Zhang, Shuai and Yao, Lina and Sun, Aixin and Tay, Yi},
  journal={ACM Computing Surveys (CSUR)},
  volume={52},
  number={1},
  pages={5},
  year={2019},
  publisher={ACM}
}
```
Thank you for your support!!!

Contributions and issues are always welcome. You can also contact me via email: cheungshuai@outlook.com

# Collaborators
[Shuai Zhang](https://sites.google.com/view/shuaizhang/home), [Yi Tay](https://sites.google.com/view/yitay/home), [Bin Wu](https://wubinzzu.github.io/)
