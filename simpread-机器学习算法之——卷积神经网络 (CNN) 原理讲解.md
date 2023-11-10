> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [zhuanlan.zhihu.com](https://zhuanlan.zhihu.com/p/156926543)

一、从神经网络到卷积神经网络
--------------

我们知道神经网络的结构是这样的：

![](https://pic3.zhimg.com/v2-35fd5c7a2b0a911a85e52559f3cd3826_r.jpg)

**那卷积神经网络跟它是什么关系呢？**

其实卷积神经网络依旧是层级网络，只是层的功能和形式做了变化，可以说是传统神经网络的一个改进。比如下图中就多了许多传统神经网络没有的层次。

![](https://pic2.zhimg.com/v2-e9a5eb5674823648ebf4c58dcca8d0ad_r.jpg)

1. 定义
-----

简而言之，卷积神经网络（Convolutional Neural Networks）是一种深度学习模型或类似于人工神经网络的多层感知器，常用来分析视觉图像。卷积神经网络的创始人是着名的计算机科学家 Yann LeCun，目前在 Facebook 工作，他是第一个通过卷积神经网络在 MNIST 数据集上解决手写数字问题的人。

![](https://pic3.zhimg.com/v2-1c25724e09bcace7fe684514ef696ffa_b.jpg)![](https://pic3.zhimg.com/v2-ba30a6edc6d4cfe533d95a2840494526_r.jpg)

2. 卷积神经网络的架构
------------

![](https://pic3.zhimg.com/v2-73af0123f03eb3fed8751bfc92d2473e_r.jpg)

如上图所示，卷积神经网络架构与常规人工神经网络架构非常相似，特别是在网络的最后一层，即全连接。此外，还注意到卷积神经网络能够接受多个特征图作为输入，而不是向量。

二、卷积网络的层级结构
-----------

一个卷积神经网络主要由以下 5 层组成：

*   数据输入层 / Input layer
*   卷积计算层 / CONV layer
*   ReLU 激励层 / ReLU layer
*   池化层 / Pooling layer
*   全连接层 / FC layer

1. 数据输入层

该层要做的处理主要是对原始图像数据进行预处理，其中包括：

*   **去均值**：把输入数据各个维度都中心化为 0，如下图所示，其目的就是把样本的中心拉回到坐标系原点上。
*   **归一化**：幅度归一化到同样的范围，如下所示，即减少各维度数据取值范围的差异而带来的干扰，比如，我们有两个维度的特征 A 和 B，A 范围是 0 到 10，而 B 范围是 0 到 10000，如果直接使用这两个特征是有问题的，好的做法就是归一化，即 A 和 B 的数据都变为 0 到 1 的范围。
*   **PCA / 白化**：用 PCA 降维；白化是对数据各个特征轴上的幅度归一化

去均值与归一化效果图：

![](https://pic1.zhimg.com/v2-4c3b00f07cce0c7dccf2e4ba5e167e30_r.jpg)

去相关与白化效果图：

![](https://pic2.zhimg.com/v2-229a2c9828a26d594dc854b659cfc8a5_r.jpg)

2. 卷积计算层

这一层就是卷积神经网络最重要的一个层次，也是 “卷积神经网络” 的名字来源。  
在这个卷积层，有两个关键操作：

*   **局部关联**。每个神经元看做一个滤波器 (filter)
*   **窗口 (receptive field) 滑动**， filter 对局部数据计算

先介绍卷积层遇到的几个名词：

*   **深度 / depth**（解释见下图）
*   **步幅 / stride** （窗口一次滑动的长度）
*   **填充值 / zero-padding**

![](https://pic3.zhimg.com/v2-821048bfaee14d8c03cc5044e04fe336_r.jpg)

还记得我们在第一篇中提到的过滤器、感受野和卷积吗？很好。现在，要改变每一层的行为，有两个主要参数是我们可以调整的。选择了过滤器的尺寸以后，我们还需要选择步幅（stride）和填充（padding）。

步幅控制着过滤器围绕输入内容进行卷积计算的方式。在第一部分我们举的例子中，过滤器通过每次移动一个单元的方式对输入内容进行卷积。过滤器移动的距离就是步幅。在那个例子中，步幅被默认设置为 1。步幅的设置通常要确保输出内容是一个整数而非分数。让我们看一个例子。想象一个 7 x 7 的输入图像，一个 3 x 3 过滤器（简单起见不考虑第三个维度），步幅为 1。这是一种惯常的情况。

![](https://pic4.zhimg.com/v2-485011463ab17396223ce79ceba0030f_r.jpg)

还是老一套，对吧？看你能不能试着猜出如果步幅增加到 2，输出内容会怎么样。

![](https://pic4.zhimg.com/v2-6b81b4032ecb28ece404e344c173481b_r.jpg)

所以，正如你能想到的，感受野移动了两个单元，输出内容同样也会减小。注意，如果试图把我们的步幅设置成 3，那我们就会难以调节间距并确保感受野与输入图像匹配。正常情况下，程序员如果想让接受域重叠得更少并且想要更小的空间维度（spatial dimensions）时，他们会增加步幅。

**填充值是什么呢？**

在此之前，想象一个场景：当你把 5 x 5 x 3 的过滤器用在 32 x 32 x 3 的输入上时，会发生什么？输出的大小会是 28 x 28 x 3。注意，这里空间维度减小了。如果我们继续用卷积层，尺寸减小的速度就会超过我们的期望。在网络的早期层中，我们想要尽可能多地保留原始输入内容的信息，这样我们就能提取出那些低层的特征。比如说我们想要应用同样的卷积层，但又想让输出量维持为 32 x 32 x 3 。为做到这点，我们可以对这个层应用大小为 2 的零填充（zero padding）。零填充在输入内容的边界周围补充零。如果我们用两个零填充，就会得到一个 36 x 36 x 3 的输入卷。

![](https://pic1.zhimg.com/v2-08fc41b67409cb6c61ecb6af72bcea30_r.jpg)

如果我们在输入内容的周围应用两次零填充，那么输入量就为 32×32×3。然后，当我们应用带有 3 个 5×5×3 的过滤器，以 1 的步幅进行处理时，我们也可以得到一个 32×32×3 的输出

如果你的步幅为 1，而且把零填充设置为

![](https://pic3.zhimg.com/v2-96a40f41090b4bb5e7ebbf0a0e186d9a_b.jpg)

K 是过滤器尺寸，那么输入和输出内容就总能保持一致的空间维度。

计算任意给定卷积层的输出的大小的公式是

![](https://pic1.zhimg.com/v2-2ea24e54873121ae877b5e7f03db8844_b.jpg)

其中 O 是输出尺寸，K 是过滤器尺寸，P 是填充，S 是步幅。

2.1 卷积的计算

（注意，下面蓝色矩阵周围有一圈灰色的框，那些就是上面所说到的填充值）

![](https://pic4.zhimg.com/v2-50e4ca38e42aa9f91e5419a107b76a07_r.jpg)

这里的蓝色矩阵就是输入的图像，粉色矩阵就是卷积层的神经元，这里表示了有两个神经元（w0,w1）。绿色矩阵就是经过卷积运算后的输出矩阵，这里的步长设置为 2。

![](https://pic1.zhimg.com/v2-af58e27acdf63325dda676ac5fc03b3c_r.jpg)

蓝色的矩阵 (输入图像) 对粉色的矩阵（filter）进行矩阵内积计算并将三个内积运算的结果与偏置值 b 相加（比如上面图的计算：2+（-2+1-2）+（1-2-2） + 1= 2 - 3 - 3 + 1 = -3），计算后的值就是绿框矩阵的一个元素。

![](https://pic2.zhimg.com/v2-3374c724f32488eb8e8552e1b9661d99_r.jpg)

下面的动态图形象地展示了卷积层的计算过程：

![](https://pic2.zhimg.com/v2-ae8a4d6f0ded77d731f179f361254db1_r.jpg)

**2.2 参数共享机制**

在卷积层中每个神经元连接数据窗的权重是固定的，每个神经元只关注一个特性。神经元就是图像处理中的滤波器，比如边缘检测专用的 Sobel 滤波器，即卷积层的每个滤波器都会有自己所关注一个图像特征，比如垂直边缘，水平边缘，颜色，纹理等等，这些所有神经元加起来就好比就是整张图像的特征提取器集合。

需要估算的权重个数减少: AlexNet 1 亿 => 3.5w

一组固定的权重和不同窗口内数据做内积: 卷积

![](https://pic1.zhimg.com/v2-32b84ca9679b8beb2692e22c0cafcb18_r.jpg)

**3. 非线性层（或激活层）**

把卷积层输出结果做非线性映射。

![](https://pic1.zhimg.com/v2-4f12096f7b6fb83ce6dc96b3ecf915c8_r.jpg)

CNN 采用的激活函数一般为 ReLU(The Rectified Linear Unit / 修正线性单元)，它的特点是收敛快，求梯度简单，但较脆弱，图像如下。更多关于激活函数的内容请看后期文章。

![](https://pic4.zhimg.com/v2-a559927aa4df378c6b1a25c2cb86db5b_r.jpg)

**激励层的实践经验：**

①不要用 sigmoid！不要用 sigmoid！不要用 sigmoid！  
② 首先试 RELU，因为快，但要小心点  
③ 如果 2 失效，请用 Leaky ReLU 或者 Maxout  
④ 某些情况下 tanh 倒是有不错的结果，但是很少

参见 Geoffrey Hinton（即深度学习之父）的论文：Rectified Linear Units Improve Restricted Boltzmann Machines **墙裂推荐此论文！** 现在上篇文章写的免费下载论文的方法就可以用上了。

**4. 池化层**

池化层夹在连续的卷积层中间， 用于压缩数据和参数的量，减小过拟合。  
简而言之，如**果输入是图像的话，那么池化层的最主要作用就是压缩图像。**

这里再展开叙述池化层的具体作用：

1.  **特征不变性**，也就是我们在图像处理中经常提到的特征的尺度不变性，池化操作就是图像的 resize，平时一张狗的图像被缩小了一倍我们还能认出这是一张狗的照片，这说明这张图像中仍保留着狗最重要的特征，我们一看就能判断图像中画的是一只狗，图像压缩时去掉的信息只是一些无关紧要的信息，而留下的信息则是具有尺度不变性的特征，是最能表达图像的特征。
2.  **特征降维**，我们知道一幅图像含有的信息是很大的，特征也很多，但是有些信息对于我们做图像任务时没有太多用途或者有重复，我们可以把这类冗余信息去除，把最重要的特征抽取出来，这也是池化操作的一大作用。
3.  在一定程度上**防止过拟合**，更方便优化。

![](https://pic2.zhimg.com/v2-deeacf1fc2ef42c0e41070fae4fb5381_r.jpg)

池化层用的方法有 Max pooling 和 average pooling，而实际用的较多的是 Max pooling。这里就说一下 Max pooling，其实思想非常简单。

![](https://pic4.zhimg.com/v2-7b28abd70e3bc4294b2b28cc6ff348ef_r.jpg)

对于每个 2 * 2 的窗口选出最大的数作为输出矩阵的相应元素的值，比如输入矩阵第一个 2 * 2 窗口中最大的数是 6，那么输出矩阵的第一个元素就是 6，如此类推。

**5. 全连接层**

两层之间所有神经元都有权重连接，通常全连接层在卷积神经网络尾部。也就是跟传统的神经网络神经元的连接方式是一样的：

![](https://pic3.zhimg.com/v2-9cbccaabf38a4c5c4c8494afc3556c12_r.jpg)

一般 CNN 结构依次为  
1. INPUT  
2. [[CONV -> RELU]N -> POOL?]M  
3. [FC -> RELU]*K  
4. FC

三、卷积神经网络的几点说明
-------------

1. 训练算法

1. 同一般机器学习算法，先定义 Loss function，衡量和实际结果之间差距。  
2. 找到最小化损失函数的 W 和 b， CNN 中用的算法是 SGD（随机梯度下降）。

2. 优缺点

（1）优点  
　　• 共享卷积核，对高维数据处理无压力  
　　• 无需手动选取特征，训练好权重，即得特征分类效果好  
（2）缺点  
　　• 需要调参，需要大样本量，训练最好要 GPU  
　　• 物理含义不明确（也就说，我们并不知道没个卷积层到底提取到的是什么特征，而且神经网络本身就是一种难以解释的 “黑箱模型”）

3. 典型 CNN

*   LeNet，这是最早用于数字识别的 CNN
*   AlexNet， 2012 ILSVRC 比赛远超第 2 名的 CNN，比
*   LeNet 更深，用多层小卷积层叠加替换单大卷积层。
*   ZF Net， 2013 ILSVRC 比赛冠军
*   GoogLeNet， 2014 ILSVRC 比赛冠军
*   VGGNet， 2014 ILSVRC 比赛中的模型，图像识别略差于 GoogLeNet，但是在很多图像转化学习问题 (比如 object detection) 上效果奇好

**4. fine-tuning**

**何谓 fine-tuning？**  
fine-tuning 就是使用已用于其他目标、预训练好模型的权重或者部分权重，作为初始值开始训练。

那为什么我们不用随机选取选几个数作为权重初始值？原因很简单，第一，自己从头训练卷积神经网络容易出现问题；第二，fine-tuning 能很快收敛到一个较理想的状态，省时又省心。

**那 fine-tuning 的具体做法是？**  
• 复用相同层的权重，新定义层取随机权重初始值  
• 调大新定义层的的学习率，调小复用层学习率

**5. 常用框架**

**Caffe**  
　• 源于 Berkeley 的主流 CV 工具包，支持 C++,python,matlab  
　•Model Zoo 中有大量预训练好的模型供使用  
  
**PyTorch**  
　•Facebook 用的卷积神经网络工具包  
　• 通过时域卷积的本地接口，使用非常直观  
　• 定义新网络层简单  
  
**TensorFlow**  
　•Google 的深度学习框架  
　•TensorBoard 可视化很方便  
　• 数据和模型并行化好，速度快

四、总结
----

卷积网络在本质上是一种输入到输出的映射，它能够学习大量的输入与输出之间的映射关系，而不需要任何输入和输出之间的精确的数学表达式，只要用已知的模式对卷积网络加以训练，网络就具有输入输出对之间的映射能力。

CNN 一个非常重要的特点就是头重脚轻（越往输入权值越小，越往输出权值越多），呈现出一个倒三角的形态，这就很好地避免了 BP 神经网络中反向传播的时候梯度损失得太快。

卷积神经网络 CNN 主要用来识别位移、缩放及其他形式扭曲不变性的二维图形。由于 CNN 的特征检测层通过训练数据进行学习，所以在使用 CNN 时，避免了显式的特征抽取，而隐式地从训练数据中进行学习；再者由于同一特征映射面上的神经元权值相同，所以网络可以并行学习，这也是卷积网络相对于神经元彼此相连网络的一大优势。卷积神经网络以其局部权值共享的特殊结构在语音识别和图像处理方面有着独特的优越性，其布局更接近于实际的生物神经网络，权值共享降低了网络的复杂性，特别是多维输入向量的图像可以直接输入网络这一特点避免了特征提取和分类过程中数据重建的复杂度。

**CNN 应用案例：**

[1] [表情识别 FER | 基于深度学习的人脸表情识别系统（Keras）](https://link.zhihu.com/?target=https%3A//blog.csdn.net/Charmve/article/details/105097315)  
[2] [表情识别 FER | 基于 CNN 分类的表情识别研究](https://link.zhihu.com/?target=https%3A//blog.csdn.net/Charmve/article/details/105097315)  
[3] [机器学习 | 卷积神经网络详解 (二)——自己手写一个卷积神经网络](https://link.zhihu.com/?target=https%3A//blog.csdn.net/Charmve/article/details/106076844)

![](https://picx.zhimg.com/v2-031f540806597266710d69834d042571_l.jpg?source=f2fdee93)Charmve14 次咨询4.9Momenta.ai Senior R&D Engineer7875 次赞同去咨询

**推荐文章**

[1] [机器学习算法之——隐马尔科夫链（Hidden Markov Models, HMM）](https://link.zhihu.com/?target=https%3A//mp.weixin.qq.com/s%3F__biz%3DMzIxMjg1Njc3Mw%3D%3D%26mid%3D2247483688%26idx%3D1%26sn%3D21a204ac6422340d251d802e9b12c4f7%26chksm%3D97befb82a0c972941c97632f04f8a34feece6dd068807a639dfc115b852be020ea5f64feeb1b%26scene%3D21%26token%3D2142822614%26lang%3Dzh_CN%23wechat_redirect)  
[2] [机器学习算法之——支持向量机（Support Vector Machine, SVM）](https://link.zhihu.com/?target=https%3A//mp.weixin.qq.com/s%3F__biz%3DMzIxMjg1Njc3Mw%3D%3D%26mid%3D2247483760%26idx%3D1%26sn%3D9fed6969a03da7821ed71aed2b647c78%26chksm%3D97befbdaa0c972ccf912b3ef138714784c70386600289d09e3dbf2393c8c5149a3eb666a2d78%26scene%3D21%26token%3D2142822614%26lang%3Dzh_CN%23wechat_redirect)  
[3] [机器学习算法之——逻辑回归 (Logistic Regression) 算法讲解及 Python 实现](https://link.zhihu.com/?target=https%3A//mp.weixin.qq.com/s%3F__biz%3DMzIxMjg1Njc3Mw%3D%3D%26mid%3D2247483865%26idx%3D1%26sn%3Dd0f7cf90b4326957ff71ec77d1d39119%26chksm%3D97befb73a0c9726504f1cf612c0c21a28f37f8f37bae61a11303a847d148497940080cd31c6c%26scene%3D21%26token%3D2142822614%26lang%3Dzh_CN%23wechat_redirect)  
[4] [机器学习算法之——梯度提升 (Gradient Boosting) 上 算法讲解及 Python 实现](https://link.zhihu.com/?target=https%3A//mp.weixin.qq.com/s%3F__biz%3DMzIxMjg1Njc3Mw%3D%3D%26mid%3D2247483916%26idx%3D1%26sn%3Db59a0d6dd31a5d5a419e6f8426f58611%26chksm%3D97bef8a6a0c971b07006603b4e3a32b009767801d33c52186cd1a6153427a1a5e1da10ffe066%26scene%3D21%26token%3D2142822614%26lang%3Dzh_CN%23wechat_redirect)  
[5] [机器学习算法之——梯度提升 (Gradient Boosting) 下 算法讲解及 Python 实现](https://link.zhihu.com/?target=https%3A//mp.weixin.qq.com/s%3F__biz%3DMzIxMjg1Njc3Mw%3D%3D%26mid%3D2247484003%26idx%3D6%26sn%3D8be6122009f862a41cc4a313c090bb0f%26chksm%3D97bef8c9a0c971dfc74b0f24a510cffee66d368689d8e97faa54e960c4b6201983b97e488fb3%26scene%3D21%26token%3D2142822614%26lang%3Dzh_CN%23wechat_redirect)  
[6] [机器学习算法之——决策树模型 (Decision Tree Model) 算法讲解及 Python 实现](https://link.zhihu.com/?target=https%3A//mp.weixin.qq.com/s%3F__biz%3DMzIxMjg1Njc3Mw%3D%3D%26mid%3D2247484003%26idx%3D6%26sn%3D8be6122009f862a41cc4a313c090bb0f%26chksm%3D97bef8c9a0c971dfc74b0f24a510cffee66d368689d8e97faa54e960c4b6201983b97e488fb3%26scene%3D21%26token%3D2142822614%26lang%3Dzh_CN%23wechat_redirect)  
[7] [机器学习算法之——K 最近邻 (k-Nearest Neighbor，KNN) 分类算法原理讲解](https://link.zhihu.com/?target=https%3A//mp.weixin.qq.com/s%3F__biz%3DMzIxMjg1Njc3Mw%3D%3D%26mid%3D2247484102%26idx%3D1%26sn%3D125c9b78013a5528fb0af1a42d01c88d%26chksm%3D97bef86ca0c9717a9347f02f9a98a816312fd7257896b22c799a0c7b6f318c57988ee6a0a68b%26scene%3D21%26token%3D2142822614%26lang%3Dzh_CN%23wechat_redirect)  
[8] [机器学习算法之——K 最近邻 (k-Nearest Neighbor，KNN) 算法 Python 实现](https://link.zhihu.com/?target=https%3A//mp.weixin.qq.com/s%3F__biz%3DMzIxMjg1Njc3Mw%3D%3D%26mid%3D2247484102%26idx%3D2%26sn%3D96cefa1a8866ed74dfdf4e054c1ba36b%26chksm%3D97bef86ca0c9717ac53d6898c35b806d21784a2118d8d70a0622e5d187f7cd8a0154beb1b0e8%26scene%3D21%26token%3D2142822614%26lang%3Dzh_CN%23wechat_redirect)