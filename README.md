# DeepRec
In this repository, a number of deep learning based recommendation models are implemented using Python and Tensorflow. We started this project in the hope that it would reduce the effects of researhers and industrial developers in reproducing state-of-the-art methods. Three major recommendation scenarios: rating prediction, top-N recommendation (item ranking) and sequential recommendation , were considered. Meanwhile, DeepRec maintains good modularity and extensibility to easily incorporate new models into the framework. It is distributed under the terms of the GNU General Public License. 

Hopefully, this repo will be useful for you. Contributions and issues are also welcome.

# Implemented Algorithms
We implemented both rating estimation, top-n recommendation models and sequence-aware recommendation models.
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

# ToDo List
* Add more models
* Different Evaluation Protocals
* Code Refactor

# Citation

To acknowledge use of this open source package in publications, please cite either of the
following papers:

```
@article{zhang2019deeprec,
  title={Deep learning based recommender system: A survey and new perspectives},
  author={Zhang, Shuai and Yao, Lina and Sun, Aixin and Tay, Yi},
  journal={ACM Computing Surveys (CSUR)},
  volume={52},
  year={2019},
  publisher={ACM}
}
```
or 
```
 @inproceedings{han2018openke,
   title={DeepRec: An Open-source Toolkit for Deep Learning based Recommendation},
   author={Shuai Zhang, Yi Tay, Lina Yao, Bin Wu, Aixin Sun},
   booktitle={Preprint},
   year={2019}
 }
```
Thank you for your support!

# Collaborators
Details of the developers can be found as follows.

[Shuai Zhang](https://sites.google.com/view/shuaizhang/home), [Yi Tay](https://sites.google.com/view/yitay/home), [Bin Wu](https://wubinzzu.github.io/)
