'''
极简版 model_type == 1
0）取消所有的signal
1)每一轮的pair 拆开分别过rnn
2）db self attn 过rnn
3）last utter 过rnn
4）decoder rnn和db weight sum
5）decoder rnn和拼接后的前轮所有表示weight sum
6）decoder rnn和last utter weight sum
7）state和三个weight sum分别残差
8）拼接word embedding
9）merge db unit表示 采用分离的方式

db rnn hid,hid
utter rnn  hid,hid
sql rnn  hid,hid
decoder
'''

'''
极简版
0）取消所有的signal
1)每一轮的pair self attn
2）db self attn
3）last utter self attn
4）decoder rnn和db weight sum
5）decoder rnn和拼接后的前轮所有表示weight sum
6）decoder rnn和last utter weight sum
7）state和三个weight sum分别残差
8）拼接word embedding
9）merge db unit表示 采用分离的方式
'''

'''
复杂版
1)每一轮的pair self attn
2）db和last utter分别和多个pair attn，并进行残差
3）cat(decoder rnn, utter rnn), 对每一个pair进行两路的attn，随后cat 
4）同上，但在utter rnn上反向再来一遍，随后两个特征拼接
5）decoder rnn和utter rnn weight sum
6）db和last utter互相attn
7）decoder rnn和db weight sum
8）decoder rnn和last utter weight sum
9）三个weight sum 跳连接
10）db合并unit
'''

'''
现在的核心问题有几个：
第一：整体的架子怎么搭建合适，是hred，还是hran，还是直接拼接？
第二：是否要引入sql
第三：decoder是否要分两路
'''

'''
公用的模块：
decoder rnn 其他都是无用的
公用的函数：
data prepare
caculate loss
atten sum
output prob
built output embedding unit
lookup from db embedding
'''

'''
第一版：
db rnn
utter 各自过rnn 并分别和db attn，db表示不动
将utter的所有表示和db的表示拼接 并decoder
'''

'''
第二版：
db rnn
utter 各自过rnn 并分别和db attn，db表示不动，同时这个过程引入另一层rnn
将utter的所有表示和db的表示拼接 并decoder
'''

'''
第三版：
db rnn
utter 各自过rnn 使用hran的方式来搞
'''
