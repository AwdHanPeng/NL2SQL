Oct29_15-59-28_ubuntu:

初步的实验观察可以发现，0-200step时，keyword已经有明显的逐步提升，逐步上升到60%，但是db unit的震荡过于严重，大部分都处于20%以下
遂考虑两点改进，
其一在decoder rnn output计算概率时，可以尝试在输出和db feature之间计算相似度量时，加入bilinear
其二在计算loss时，在融合db loss和keyword loss时加入权重，可以尝试的权重是除以各自个valid step
此外，模型会再学习一定阶段，出现loss nan的情况
考虑进行实验完善：
1 decoder output 首先和输入的表示进行拼接  转变维度之后过非线性层
2 非线性层之后的结果，在与计算db表示进行attn时，需要进行维度转换操作


Oct29_19-17-39_ubuntu:
完善了三个部分：
1，在session的每一轮前加入了pre turn content，这个content内部只有db，因此这个content可以用来学习生成第一个sql语句
2，加入了decoder输入与输出的拼接模块，并将两者过tanh融合，随后再计算概率
3，再计算db unit概率时，额外引入一个线性层用于将输出表示转换空间，随后再与db表示进行相似度计算
实验结果：
1，在已有的结果内，模型的key loss学习的更快，
2，模型的db unit loss震荡问题仍然没有得到解决
3，模型会更快的到达nan的情况 从1.2k迁移到500k
考虑进行实验完善：
针对nan的问题，考虑梯度夹紧，因为有两层的rnn，所以必须考虑梯度的夹紧问题


Oct29_19-58-43_ubuntu:
完善了一个部分：
1，添加了对非bert参数的grad的夹紧
结果：
从目前的结果来看，对db会略有提升
考虑对bert的参数也进行梯度夹紧 或者直接fix 因为有点担心是bert的梯度爆炸了

Oct29_20-12-27_ubuntu:
完善了一个部分：
1，将bert参数进行fix
结果：
从前200个样本的效果来看，差别并不是特别大，因此建议调试时固定bert为准
进一步的任务目标，首先观察梯度夹紧是否能解决nan的问题，并搞清楚为什么db unit学习不动
首先观察第500个样本左右是否出现问题：实验发现，clip的确是有效果的，在特定位置没有再出现nan的现象
针对db unit学习的问题，拟采用措施：
1，去除keyword loss
2，对db loss加入loss相关的权重


Oct29_21-20-14_ubuntu:
尝试了一个改动，仅仅选取了db unit loss进行学习
1，去除keyword loss
显示结果是 训练loss仍然震荡，但是考虑这仅仅是训练集的问题，所以对实验进行回退
并且考虑到log的众生平等效果，所以应该不是loss的原因，而且也不一定是模型出现了问题
还是要综合观察验证集的综合表现，设置weight的想法可以暂时不必考虑


Oct29_22-10-05_ubuntu:
此次实验的设置是简单相加，并长期跑下去以观察验证集的表现


查阅了editsql的log日志，确认其valid gold-passing TOKEN_ACCURACY就是teacher forcing情况下的token acc
editsql的数值最高时90.6 那么只要我们在验证集上超过了这个数值  那么我们就sota了

突然反应过来，如果考虑计算string acc，也就是只要有一个token不对，那么这句话就是错的
那么这样的要求下，teacher forcing和非teacher force其实是完全一样的
妙啊！！！
完全不需要写迭代式生成的函数 就可以进行测量了,这个数值是56，

然后还需要重点解决仍然会出现的查不到的情况

之后的改进方向应该放在多头db的表示上！
可能是这个地方引入了太多的噪音了 mask掉才应该（这个地方本来就是加的0，mask也是无意义的）

Oct30_00-18-44_ubuntu:
此次实验的设置是通过utter level的mask，去除不必要的db表示，
并首先保留了拼接的方式
在前200个step没有发现明显的效果改进（改进就是有鬼了）


Oct30_00-34-47_ubuntu:
此次实验设置对训练集进行shuffle，观察效果如何
也可尝试先计算turn多的样本
在前200个step中，发现训练效果和之前没有明显区别

Oct30_01-00-59_ubuntu:
此次实验摒弃数据打乱的做法，并去除db concat，采用add的方式
发现并没有明显的效果改进 （甚至会降低）

考虑是不是transformer的原因？
transformer堆叠的层数过多，有可能导致这个问题

Oct30_01-22-06_ubuntu:
此次实验将transformer层数降低，发现训练效果并没有明显差异（有差异，层数降低后loss变大）

Oct30_02-11-18_ubuntu:
采用5条数据进行拟合，发现依然震荡

Oct30_02-25-59_ubuntu:
采用随机20条数据进行拟合，发现依然震荡，并且出现nan现象
在调入了99epoch的模型数据后，发现signal embedding全部变成nan，rnn cell也变成nan
随考虑参考editsql中加入噪声的做法，并查阅了文档，决定将softmax转换为log softmax

Oct30_11-54-46_ubuntu:
此次实验将softmax转换为log softmax
从实验效果来看彻底解决掉了nan问题，模型能够继续学下去（好的改进）
但是学习依然震荡，key acc上限0.6，db acc在0.6和0.2之间来回震荡
实验目标：考察梯度夹紧是否真的有作用，可以考虑去除掉
然后去除warm up并设置低一个数量级的lr，从1e-4改变到1e-5

Oct30_12-58-25_ubuntu:
去除了梯度clip
然后去除warm up并设置低一个数量级的lr，从1e-4改变到1e-5
实验结果发现db的学习有明显改观，key的学习稍有进步，但上限仍然在0.6左右 （好的改进）
说明学习率在1e-5的数量级是更好的
但是两个acc的上限仍然在0.6左右 模型仍然没有快速过拟合
说明模型还是有问题

依照上述模型的参数配置，进行debug


Oct30_19-26-11_ubuntu:
检查发现在合并db表示时，从dbfeature中的idx计算错误
此外去除了不必要的mask，因为前边伪表示都是拼接的0单元，所以没有必要再mask置为0了
目前检查到hie decoder部分
尝试进行实验
实验结果：发现居然并没有任何明显的提升，模型还是不怎么能过拟合的样子，看样子模型还是有问题

Oct30_21-58-14_ubuntu:
检查hie decoder部分，在rnn hidd初始化时，进行了mask的操作,避免一股脑的sum
而且为了防止出现一句话中全是pad，仅仅全部设置为mask，在计算atten时仍然会得到一个sum，所以设置了hard atten，在attn之前就将mask位置的value设置成0
而且需要注意一点，就算给bert输入的某个token是mask过的，这个token过完bert并不是全零，因为bert有位置编码
所以才在计算atten时采用了hard的mask形式，避免出现之后的全mask即avg的现象
模型其他部分全部检查完毕
初次实验发现会超GPU内存，和之前有区别 所以不得已只好将transformer降到2层


下一步的改进措施：因为sql关键字根本就没有一个好的初始化的表示，所以是有问题的，考虑两种方案，
一是用bert初始化一个然后切断和bert之间的梯度，
另一个就是采用拼接的方式

