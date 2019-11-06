# Adversarial-Preference-Learning-with-Pairwise-Comparisons
This is a Keras implementation of the model described in our paper:

>Z. Wang, Q. Xu, K. Ma, Y. Jiang, X. Cao and Q. Huang. Adversarial Preference Learning with Pairwise Comparisons. MM2019.

## Dependencies
- Keras >= 2.2.4
- Tensorflow >= 1.12.0
- numpy

## Data
We convert the datasets `ML100K`, `ML1M` and `Netflix` to our train and test files in the `data/` folder. 

For each user, we randomly select $N=50$ items to generate pairwise comparisons and store the train data in the files **.dat*. The data format is *(user_id, item_id_1, item_id_2)*, which means the user prefers the item1 than the item2. The ids start from 1.

The rest ratings are treated as test data and stored in the files *.lsvm. The data format of the line *user_id* is *item_id:rating*. The numeric ratings range from 1 to 5.

## Train
Here is an example to train the model with logistic loss.
```
python cr_gan.py
```

## Citation
Please cite our paper if you use this code in your own work:

```
@inproceedings{wang2019Adversarial,
  title={Adversarial Preference Learning with Pairwise Comparisons},
  author={Wang, Zitai and Xu, Qianqian and Ma, Ke and Jiang, Yangbangyan and Cao, Xiaochun and Huang, Qingming},
  booktitle={ACM on Multimedia Conference},
  pages={656--664},
  year={2019}
}
```
