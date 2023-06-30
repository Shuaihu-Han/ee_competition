# Requirements

We conduct our experiments on the following environments:

```
python 3.6
CUDA: 9.0
GPU: Tesla T4
pytorch == 1.1.0
transformers == 4.23.1
```

# How to run

1.训练trigger:

```
python main4trigger.py
```
2.训练arg:

```
python main4arg.py
```
3.评估模型结果:

```
python evaluate_staged.py
```

The hyper-parameters are recorded in ``/utils/params4trigger.py`` for ``main4trigger.py`` and in ``/utils/params4arg.py`` for ``main4arg.py``. 
We adopt ``bert-base-chinese`` as our pretrained language model. For extention, you could also try further hyper-parameters for even better performance.
当前模型在``main4trigger``训练2个epoch以及``main4arg.py``训练10个epoch后, 可以达到47.4的准确率.

# Citation

If you find this code useful, please cite our work:

```
@inproceedings{Sheng2021:CasEE,
    title = "{C}as{EE}: {A} Joint Learning Framework with Cascade Decoding for Overlapping Event Extraction",
    author = "Sheng, Jiawei and
      Guo, Shu and
      Yu, Bowen and
      Li, Qian and
      Hei, Yiming and
      Wang, Lihong and
      Liu, Tingwen and
      Xu, Hongbo",
    booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-acl.14",
    doi = "10.18653/v1/2021.findings-acl.14",
    pages = "164--174",
}
```

