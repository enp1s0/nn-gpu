# ニューラルネットワーク

## 実行環境
- NVIDIAのGPUが欲しい
- C++11がコンパイルできて欲しい

## 命名規則
- 基本適当
- 行列の入出力のある関数の引数は(output,input,...)の順

### 実行ログ
10000 calc
#### P100
```
network0(784,225,32)
 - learning rate = 0.1
 - adagrad epsilon = 1
 - momentum rate = 0.2
network1(225,10,32)
 - learning rate = 0.1
 - adagrad epsilon = 1
 - momentum rate = 0.2
Loading training data ... DONE : 1779.04 [ms]
Start training
1000 / 10000 (10%)
2000 / 10000 (20%)
3000 / 10000 (30%)
4000 / 10000 (40%)
5000 / 10000 (50%)
6000 / 10000 (60%)
7000 / 10000 (70%)
8000 / 10000 (80%)
9000 / 10000 (90%)
10000 / 10000 (100%)
Done : 104451 [ms]
```

#### V100
```
network0(784,225,32)
 - learning rate = 0.1
 - adagrad epsilon = 1
 - momentum rate = 0.2
network1(225,10,32)
 - learning rate = 0.1
 - adagrad epsilon = 1
 - momentum rate = 0.2
Loading training data ... DONE : 1569.03 [ms]
Start training
1000 / 10000 (10%)
2000 / 10000 (20%)
3000 / 10000 (30%)
4000 / 10000 (40%)
5000 / 10000 (50%)
6000 / 10000 (60%)
7000 / 10000 (70%)
8000 / 10000 (80%)
9000 / 10000 (90%)
10000 / 10000 (100%)
Done : 87290.8 [ms]
```

## MNIST学習結果
![](./result-matrix.png)
