# 2021-CPS CVPR

> 论文题目：Semi-Supervised Semantic Segmentation with Cross Pseudo Supervision
>
> 中文题目：基于交叉伪监督的半监督语义分割
>
> 论文链接：[https://arxiv.org/abs/2106.01226v2](https://arxiv.org/abs/2106.01226v2)
>
> 论文代码：[https://github.com/charlesCXK/TorchSemiSeg](https://github.com/charlesCXK/TorchSemiSeg)
>
> 发表时间：2021年6月
>
> 引用：Chen X, Yuan Y, Zeng G, et al. Semi-supervised semantic segmentation with cross pseudo supervision[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021: 2613-2622.
>
> 引用数：157



## 1. 简介



### 1.1 简介

在这篇论文中，我们为半监督语义分割任务设计了一种**非常简洁而又性能很好**的算法：cross pseudo supervision (CPS)。训练时，我们使用两个相同结构、但是不同初始化的网络，添加约束使得**两个网络对同一样本的输出是相似的**。具体来说，当前网络产生的one-hot pseudo label，会作为另一路网络预测的目标，这个过程可以用cross entropy loss监督，就像传统的全监督语义分割任务的监督一样。我们在两个benchmark (PASCAL VOC, Cityscapes) 都取得了SOTA的结果。



### 1.2 相关工作

在最开始，我们先来回顾一下半监督语义分割任务的相关工作。不同于图像分类任务，数据的标注对于语义分割任务来说是比较困难而且成本高昂的。我们需要为图像的每一个像素标注一个标签，包括一些特别细节的物体，比如下图中的电线杆 (Pole)。但是，我们可以很轻松的获得RGB数据，比如摄像头拍照。那么，如何利用大量的无标注数据去提高模型的性能，成为半监督语义分割领域研究的问题。

![img](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/v2-8619394ff5baabce776e6265ff37e684_720w.webp)

我们将半监督分割的工作总结为两种：self-training和consistency learning。一般来说，self-training是离线处理的过程，而consistency learning是在线处理的。

**（1）Self-training**

Self-training主要分为3步。第一步，我们在有标签数据上训练一个模型。第二步，我们用预训练好的模型，为无标签数据集生成伪标签。第三步，使用有标注数据集的真值标签，和无标注数据集的伪标签，重新训练一个模型。

![img](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/v2-51ea67f1ae6deb8ed92148581cfc85a7_720w.webp)

**（2）Consistency learning**

Consistency learning的核心idea是：**鼓励模型对经过不同变换的同一样本有相似的输出**。这里“变换”包括高斯噪声、随机旋转、颜色的改变等等。

Consistency learning基于两个假设：smoothness assumption 和 cluster assumption。

- **Smoothness assumption**: samples close to each other are likely to have the same label.
- **Cluster assumption**: Decision boundary should lie in low-density regions of the data distribution.

Smoothness assumption就是说靠的近的样本通常有相同的类别标签。比如下图里，蓝色点内部距离小于蓝色点和棕色点的距离。Cluster assumption是说，模型预测的决策边界，通常处于样本分布密度低的区域。怎么理解这个“密度低”？我们知道，类别与类别之间的区域，样本是比较稀疏的，那么一个好的决策边界应该尽可能处于这种样本稀疏的区域，这样才能更好地区分不同类别的样本。例如下图中有三条黑线，代表三个决策边界，实线的分类效果明显好于另外两条虚线，这就是处于低密度区域的决策边界。

![img](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/v2-b48a81617699b0d8598ca7685ce9fb8a_720w.webp)

那么，**consistency learning是如何提高模型效果的呢**？在consistency learning中，我们通过对一个样本进行扰动（添加噪声等等），即改变了它在feature space中的位置。但我们希望模型对于改变之后的样本，预测出同样的类别。这个就会导致，在模型输出的特征空间中，同类别样本的特征靠的更近，而不同类别的特征离的更远。只有这样，扰动之后才不会让当前样本超出这个类别的覆盖范围。这也就导致学习出一个更加compact的特征编码。

当前，Consistency learning主要有三类做法：mean teacher，CPC，PseudoSeg。

![img](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/v2-7ce92479927757cad5015da5ffb2c7b7_720w.webp)

Mean teacher是17年提出的模型。给定一个输入图像X，添加不同的高斯噪声后得到X1和X2。我们将X1输入网络f(θ)中，得到预测P1；我们对f(θ)计算EMA，得到另一个网络，然后将X2输入这个EMA模型，得到另一个输出P2。最后，我们用P2作为P1的目标，用MSE loss约束。



![img](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/v2-7abbc69261b0407de12b105533c83e0d_720w.webp)

PseudoSeg是google发表在ICLR 2021的工作。他们对输入的图像X做两次不同的数据增强，一种“弱增强”（random crop/resize/flip），一种“强增强”(color jittering)。他们将两个增强后图像输入同一个网络f(θ)，得到两个不同的输出。因为“弱增强”下训练更加稳定，他们用“弱增强”后的图像作为target。



![img](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/v2-8c3a1f2b87e78bb08d2a2295e1fb8ef3_720w.webp)

CPC是发表在ECCV 2020的工作（Guided Collaborative Training for Pixel-wise Semi-Supervised Learning）的**简化版本**。在这里，我只保留了他们的核心结构。他们将同一图像输入两个不同网络，然后约束两个网络的输出是相似的。这种方法虽然简单，但是效果很不错。





### 1.3 创新

从上面的介绍我们可以简单总结一下：

- Self-training可以通过pseudo labelling扩充数据集。
- CPC可以通过consistcency learning，鼓励网络学习到一个更加compact的特征编码。

大家近年来都focus在consistency learning上，而忽略了self-training。实际上，我们实验发现，self-training在数据量不那么小的时候，性能非常的强。那么我们很自然的就想到，为什么不把这两种方法结合起来呢？于是就有了我们提出的CPS：cross pseudo supervision。





## 2. 网络



### 2.1 网络架构

![image-20221120194641900](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/image-20221120194641900.png)

> * a) 本论文网络结构图CPS，叙述如上
> * b) GCT网络结构图，和本方法相同，唯一的区别在于GCT使用confidence map作为监督信号，而CPS使用one-hot label
> * c）MeanTeacher 使用学生网络和老师网络，两个网络结构相同，参数初始化不同，学生网络用老师网络得到的confidence map作为监督信号，老师网络随着学生网络的权重变化按照指数平均不断变化
> * d）一张原图X分别经过弱数据增强和强数据增强放入同一个网络中，弱数据增强所得one hot结果作为强数据增强结果的真值，用于监督弱数据增强的结果

我们可以看到，CPS的设计非常的简洁。训练时，我们使用两个网络f(θ1) 和 f(θ2)。这样对于同一个输入图像X，我们可以有两个不同的输出P1和P2。我们通过argmax操作得到对应的one-hot标签Y1和Y2。类似于self-training中的操作，我们将这两个伪标签作为监督信号。举例来说，我们用Y2作为P1的监督，Y1作为P2的监督，并用cross entropy loss约束。

对于这两个网络，我们使用相同的结构，但是不同的初始化。我们用PyTorch框架中的kaiming_normal进行两次随机初始化，而没有对初始化的分布做特定的约束。



### 2.2 Loss

> 给定有标签的数据集$D^L$，大小为$N$ 。和$M$个未标记的图像$D^u$
>
> 半监督语义分割任务的目标是通过对标记图像和未标记图像来学习分割网络

$$
\begin{array}{l}
P_{1}=f\left(X ; \theta_{1}\right) \\
P_{2}=f\left(X ; \theta_{2}\right)
\end{array}
$$

这两个网络具有相同的结构,$\theta_{1}$和$\theta_{2}$分别表示对应的权重，初始化的方式不同。输入X具有相同的数据增强方式，P1 ，P2为分割confidence map，为softmax归一化后的网络输出。本文的主要思想通过以下的方式表达：
$$
\begin{array}{l}
X \rightarrow \mathbf{X} \rightarrow f\left(\theta_{1}\right) \rightarrow P_{1} \rightarrow Y_{1} \\
X \rightarrow \mathbf{X} \rightarrow f\left(\theta_{2}\right) \rightarrow P_{2} \rightarrow Y_{2}
\end{array}
$$
![image-20221120205427625](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/image-20221120205427625.png)

$P_1,P_2$表示预测结果。$Y_1,Y_2$伪分割图

流程上图所示。训练过程包括`两个损失`：`监督损失`$L_s$,`交叉伪监督损失`$L_{cps}$

监督损失使用Cross entropy。

$L_s$是两个模型的`有监督损失`
$$
\begin{array}{r}
\mathcal{L}_{s}=\frac{1}{\left|\mathcal{D}^{l}\right|} \sum_{\mathbf{X} \in \mathcal{D}^{l}} \frac{1}{W \times H} \sum_{i=0}^{W \times H}\left(\ell_{c e}\left(\mathbf{p}_{1 i}, \mathbf{y}_{1 i}^{*}\right)\right. 
\left.+\ell_{c e}\left(\mathbf{p}_{2 i}, \mathbf{y}_{2 i}^{*}\right)\right),
\end{array}
$$

----

`交叉伪监督损失`是双向的:一个是从$f(\theta_1)$到$f(\theta_2)$，使用$Y_1$来监督$P_2$,另一个是使用$Y_2$监督$P_1$。损失表示为
$$
\begin{array}{l}
\mathcal{L}_{c p s}^{u}=\frac{1}{\left|\mathcal{D}^{u}\right|} \sum_{\mathbf{X} \in \mathcal{D}^{u}} \frac{1}{W \times H} \sum_{i=0}^{W \times H}\left(\ell_{c e}\left(\mathbf{p}_{1 i}, \mathbf{y}_{2 i}\right)\right.
\left.+\ell_{c e}\left(\mathbf{p}_{2 i}, \mathbf{y}_{1 i}\right)\right) . \\
\end{array}
$$
以同样的方式定义了有标记数据上的交叉伪监督损失.$L_{cps}^l,L_{cps}=L_{cps}^l+L_{cps}^u$



最后损失为$L=L_s+L_{cps}$



**应用CutMix augmentation图像增强方法来进行数据增强。**



### 2.3 实验

> a，b图分别使用ResNet50，ResNet101作为backbone
> 蓝：baseline（不加入cutmix数据增强）：只使用有标签的数据
> 红：使用两个网络，半监督学习，不加入cutmix数据增强
> 绿：使用两个网络，半监督学习，加入cutmix数据增强

![image-20221120194524728](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/image-20221120194524728.png)

> 首先是有标签数据比较少的情况。
>
> 我们的方法在VOC和Cityscapes两个数据集的几种不同的数据量情况下都达到了SOTA。表格中 1/16, 1/4等表示用原始训练集的 1/16, 1/4作为labeled set，剩余的 15/16, 3/4作为unlabeled set。
>
> Table1:在Pascal VOC上使用不同backbone和不同有标签数据比例得到结果和其他SOTA方法对比

![image-20221120194549071](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/image-20221120194549071.png)

> Table2:在Cityscapes上使用不同backbone和不同有标签数据比例得到结果和其他SOTA方法对比

![image-20221120194609128](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/image-20221120194609128.png)

> 在跟PseudoSeg的对比中，和他们同样的数据划分list，我们也超越了他们的性能：

![img](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/v2-fe5af2e767164749359160d9f8b0d59d_720w.webp)

> 这是我们的方法跟self-training进行比较的结果。可以看到，我们的方法由于鼓励模型学习一个更加compact的特征编码，显著地优于self-training。

![img](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/v2-aa7eff874e4f5629c1ccef5abf3fa426_720w.webp)



> 对于这两个网络，我们使用相同的结构，但是不同的初始化。我们用PyTorch框架中的kaiming_normal进行两次随机初始化，而没有对初始化的分布做特定的约束。

![img](https://resource-joker.oss-cn-beijing.aliyuncs.com/picture/v2-a238711bedf5cc526a1ae3cd0c7c53aa_720w.webp)



## 3. 代码

~~~python

~~~



参考资料

[[CVPR 2021\] CPS: 基于交叉伪监督的半监督语义分割 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/378120529)