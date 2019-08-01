# DeepRec
In this repository, a number of deep learning based recommendation models are implemented using Python and Tensorflow. 
We started this project in the hope that it would reduce the efforts of researchers and developers in reproducing state-of-the-art methods. The implemented models cover three major recommendation scenarios: rating prediction, top-N recommendation (i.e., item ranking) and sequential recommendation. Meanwhile, DeepRec maintains good modularity and extensibility for easy incorporation of new models into this framework. DeepRec is distributed under the GNU General Public License.


Anyone who is interested in contributing to this project, please contact me!


# Algorithms Implemented
We implemented both rating estimation, top-n recommendation models and sequence-aware recommendation models.
* I-AutoRec and U-AutoRec (www'15)
* CDAE (WSDM'16)
* NeuMF (WWW'17)
* CML (WWW'17)
* LRML (WWW'18) (DRAFT version)
* NFM (SIGIR'17)
* NNMF (arxiv)
* PRME (IJCAI 2015)
* CASER (WSDM 2018)
* AttRec (AAAI 2019 RecNLP)
and so on.

To use the code, run: Test/test_item_ranking.py, Test/test_rating_pred.py, or Test/testSeqRec.py

# Requirements
* Tensorflow 1.7+, Python 3.5+, numpy, scipy, sklearn, pandas

# ToDo List
* More deep-learning based models
* Alternative evaluation protocols
* Code refactoring
* Update to Tensorflow 2.0

# Citation

To acknowledge use of this open source package in publications, please cite either of the following papers:

```
@article{zhang2019deeprecsyscsur,
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
 @inproceedings{shuai2019deeprec,
   title={DeepRec: An Open-source Toolkit for Deep Learning based Recommendation},
   author={Shuai Zhang, Yi Tay, Lina Yao, Bin Wu, Aixin Sun},
   journal={arXiv preprint arXiv:1905.10536},
   year={2019}
 }
```
Thank you for your support!

The chinese version is host [here](https://code.ihub.org.cn/projects/274).
