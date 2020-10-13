### blog: [https://amberzzzz.github.io/2020/09/25/metric-learning%E7%B3%BB%E5%88%97/]

1. siamese & contrastive loss
    shared cnn
    ap & an pairs各占一半，生成pairs贼慢
    siamese论文是cosine distance，contrastive loss论文是euclidean distance
    optimize target: disp->0, disn>m
    尝试设置m=1和2，对结果影响不大，test center acc=0.981


2. facenet & triplet-loss
    a-p-n pairs
    online triplet selection within mini-batch: keep all aps & hard ans
    l2 norm, l2 distance, 
    每个mini-batch要各类别均匀采样
    训练不收敛


3. center-loss
    embedding version不太好收敛，scale不好调
    在softmax分类头上test acc早就1了，
    用metric和质心度量类别准确率只有0.99，因为center-loss只关注类内距离
    两个超参：remains stable across a large range，论文没有给出最佳／建议value，lambda 0-0.1，alpha 0.01-1


4. triplet-center-loss
    是center-loss的补充，类内基础上再加上类间，
    其中类间选用距离最小的簇心距离
    改善了center-loss画图不好看的问题，joint supervision主要还是靠softmax头


5. circle-loss
    circle loss的输出也是embedding，然后通过计算和每个类别簇心的cosine distance得到类别
    optimize target: simp->1+m & simn->-m
    是目前实验下来用embedding度量类别里面结果最好的，test center acc=1.0
    而且N -> N\*N的计算（case to pair）在网络里面，比较高效


6. vis
    用TSNE降维高维的embedding到2d上，不全可分的，没论文好看
    直接训2d embedding可能比较好看


