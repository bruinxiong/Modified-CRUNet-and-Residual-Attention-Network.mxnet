import mxnet as mx
import symbol_utils
# Coded by Lin Xiong on Aug-15, 2018
# Modified from pytorch code https://github.com/tengshaofeng/ResidualAttentionNetwork-pytorch, 
# More detailed information can be found in the following paper:
# Fei Wang, Mengqing Jiang, Chen Qian, Shuo Yang, Cheng Li, Honggang Zhang, Xiaogang Wang and Xiaoou Tang, "Residual Attention Network for Image Classification", CVPR 2017 Spotlight, https://arxiv.org/pdf/1704.06904.pdf
# We also refer the input setting of this paper II:
# Jiankang Deng, Jia Guo and Stefanos Zafeiriou, "ArcFace: Additive Angular Margin Loss for Deep Face Recognition", arXiv:1801.o7698v1
# The size of input faces is only 112x112 not 224x224

bn_mom = 0.9

# Basic layers
def BN(data, momentum=bn_mom, fix_gamma=False, eps=2e-5, name=None, suffix=''):
    bn = mx.sym.BatchNorm(data=data, name='%s%s_batchnorm' %(name, suffix), fix_gamma=fix_gamma, momentum=momentum, cudnn_off=True)
    # bn = mx.sym.BatchNorm(data=data, name='%s%s_batchnorm' %(name, suffix), fix_gamma=fix_gamma, eps=eps, momentum=momentum)
    return bn

def Act(data, act_type='prelu', name=None):
    body = mx.sym.LeakyReLU(data = data, act_type=act_type, name = '%s_%s' %(name, act_type))
    #body = mx.sym.Activation(data=data, act_type='relu', name=name)
    return body

def Conv(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, workspace=256, name=None, w=None, b=None, suffix=''):
    if w is None:
        conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, num_group=num_group, stride=stride, pad=pad, no_bias=True, workspace=workspace, name='%s%s_conv2d' %(name, suffix))
    else:
        if b is None:
            conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, num_group=num_group, stride=stride, pad=pad, no_bias=True, workspace=workspace, weight=w, name='%s%s_conv2d' %(name, suffix))
        else:
            conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, num_group=num_group, stride=stride, pad=pad, workspace=workspace, weight=w, bias=b, name='%s%s_conv2d' %(name, suffix))
    return conv

def BN_Act(data, momentum=bn_mom, name=None, suffix=''):
    bn = BN(data, momentum=momentum, fix_gamma=False, eps=2e-5, name=name, suffix=suffix)
    bn_act = Act(bn, act_type='prelu', name=name)
    return bn_act

def BN_Conv(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, momentum=bn_mom, workspace=256, name=None, w=None, b=None, suffix=''):
    bn = BN(data, momentum=momentum, fix_gamma=False, eps=2e-5, name=name, suffix=suffix)
    bn_conv = Conv(bn, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, num_group=num_group, workspace=workspace, name=name, w=w, b=b, suffix=suffix)
    return bn_conv

def BN_Act_Conv(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, momentum=bn_mom, workspace=256, name=None, w=None, b=None, suffix=''):
    bn = BN(data, momentum=momentum, fix_gamma=False, eps=2e-5, name=name, suffix=suffix)
    bn_act = Act(bn, act_type='prelu', name=name)
    bn_act_conv = Conv(bn_act, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, num_group=num_group, workspace=workspace, name=name, w=w, b=b, suffix=suffix)
    return bn_act_conv

# IBN block
def IBN_block(data, num_filter, name, eps=2e-5, bn_mom=0.9, suffix=''):
    split = mx.symbol.split(data=data, axis=1, num_outputs=2)
    # import pdb
    # pdb.set_trace()
    out1 = mx.symbol.InstanceNorm(data=split[0], eps=eps, name=name + '_ibn' + '_in1')
    out2 = BN(split[1], momentum=bn_mom, fix_gamma=False, eps=eps, name=name + '_ibn', suffix=suffix)
    out = mx.symbol.Concat(out1, out2, dim=1, name=name + '_ibn1')
    return out

# Residual block (According to improved residual block from paper II: ArcFace: Additive Angular Margin Loss for Deep Face Recognition)
def Residual_unit(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs):
    bn_mom = kwargs.get('bn_mom', 0.9)
    workspace = kwargs.get('workspace', 256)
    eps = kwargs.get('eps', 2e-5)
    use_se = kwargs.get('version_se', 1)
    act_type = kwargs.get('version_act', 'prelu')
    ibn = kwargs.get('ibn', False)
    memonger = kwargs.get('memonger', False)

    if bottle_neck:
        if num_filter == 2048:
          ibn = False
        if ibn:
          bn1 = IBN_block(data=data, num_filter=int(num_filter*0.25), name='%s_c1x1' %(name))
        else:
          bn1 = BN(data, momentum=bn_mom, fix_gamma=False, eps=eps, name='%s_c1x1' %(name), suffix='')
        # act1 = Act(bn1, act_type=act_type, name='%s_c1x1' %(name))
        conv1 = Conv(bn1, num_filter=int(num_filter*0.25), kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, workspace=workspace, name='%s_c1x1_a' %(name), suffix='')
        conv2 = BN_Act_Conv(conv1, num_filter=int(num_filter*0.25), kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=1, 
                                         momentum=bn_mom, workspace=workspace, name='%s_c3x3' %(name))
        conv3 = BN_Act_Conv(conv2, num_filter=num_filter, kernel=(1, 1), stride=stride, pad=(0, 0), num_group=1,
                                         momentum=bn_mom, workspace=workspace, name='%s_c1x1_b' %(name))
        conv3 = BN(conv3, momentum=bn_mom, fix_gamma=False, eps=eps, name='%s_bn_c1x1_b' %(name))

        if use_se:
            #se begin, in original paper, excitation part is implemented by Fully connected layer not Convolution layer in here. The purpose comes from economic parameter consumption.
            body = mx.sym.Pooling(data=conv3, global_pool=True, kernel=(7, 7), pool_type='avg', name='%s_se_pool1' %(name))
            body = Conv(body, num_filter=num_filter//16, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, workspace=workspace, name='%s_se_1' %(name), suffix='')
            body = Act(body, act_type=act_type, name='%s_se' %(name))
            body = Conv(body, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, workspace=workspace, name='%s_se_2' %(name), suffix='')
            body = mx.symbol.Activation(data=body, act_type='sigmoid', name='%s_se_sigmoid' %(name))
            conv3 = mx.symbol.broadcast_mul(conv3, body)
            #se end

        if dim_match:
            shortcut = data
        else:
            conv1sc = Conv(data, num_filter=num_filter, kernel=(1, 1), stride=stride, pad=(0, 0), num_group=1, workspace=workspace, name='%s_conv1sc' %(name), suffix='')
            shortcut = BN(conv1sc, momentum=bn_mom, fix_gamma=False, eps=eps, name='%s_bn_sc' %(name))
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return conv3 + shortcut
    else:
        if num_filter == 512:
          ibn = False
        if ibn:
            bn1 = IBN_block(data=data, num_filter=num_filter, name='%s_c3x3' %(name))
        else:
            bn1 = BN(data, momentum=bn_mom, fix_gamma=False, eps=eps, name='%s_c3x3' %(name), suffix='')
        conv1 = Conv(bn1, num_filter=num_filter, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=1, workspace=workspace, name='%s_c3x3_a' %(name), suffix='')
        conv2 = BN_Act_Conv(conv1, num_filter=num_filter, kernel=(3, 3), stride=stride, pad=(1, 1), num_group=1, 
                                         momentum=bn_mom, workspace=workspace, name='%s_c3x3_b' %(name))
        conv2 = BN(conv2, momentum=bn_mom, fix_gamma=False, eps=eps, name='%s_bn_c3x3_b' %(name))

        if use_se:
            #se begin, in original paper, excitation part is implemented by Fully connected layer not Convolution layer in here. The purpose comes from economic parameter consumption.
            body = mx.sym.Pooling(data=conv3, global_pool=True, kernel=(7, 7), pool_type='avg', name='%s_se_pool1' %(name))
            body = Conv(body, num_filter=num_filter//16, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, workspace=workspace, name='%s_se_1' %(name), suffix='')
            body = Act(body, act_type=act_type, name='%s_se' %(name))
            body = Conv(body, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, workspace=workspace, name='%s_se_2' %(name), suffix='')
            body = mx.symbol.Activation(data=body, act_type='sigmoid', name='%s_se_sigmoid' %(name))
            conv2 = mx.symbol.broadcast_mul(conv2, body)
            #se end

        if dim_match:
            shortcut = data
        else:
            conv1sc = Conv(data, num_filter=num_filter, kernel=(1, 1), stride=stride, pad=(0, 0), num_group=1, workspace=workspace, name='%s_conv1sc' %(name), suffix='')
            shortcut = BN(conv1sc, momentum=bn_mom, fix_gamma=False, eps=eps, name='%s_bn_sc' %(name))
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return conv2 + shortcut

# AttentionModule_stage1: input size is 56x56
def AttentionModule_stage1(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs):
    size1 = kwargs.get('size1', 56)
    size2 = kwargs.get('size2', 28)
    size3 = kwargs.get('size3', 14)
    bn_mom = kwargs.get('bn_mom', 0.9)
    workspace = kwargs.get('workspace', 256)
    eps = kwargs.get('eps', 2e-5)
    version_se = kwargs.get('version_se', 1)
    act_type = kwargs.get('version_act', 'prelu')
    ibn = kwargs.get('ibn', False)
    memonger = kwargs.get('memonger', False)

    # Related with p parameter of paper, p=1
    RB_first = Residual_unit(data=data, num_filter=num_filter, stride=stride, dim_match=dim_match, name='%s_rb_first' %(name), bottle_neck=bottle_neck, **kwargs)
    # Trunk branch begin
    # Related with t parameter of papar, t=2
    Trunk_RB_1 = Residual_unit(data=RB_first, num_filter=num_filter, stride=stride, dim_match=dim_match, name='%s_tk_rb_1' %(name), bottle_neck=bottle_neck, **kwargs)
    Trunk_RB_2 = Residual_unit(data=Trunk_RB_1, num_filter=num_filter, stride=stride, dim_match=dim_match, name='%s_tk_rb_2' %(name), bottle_neck=bottle_neck, **kwargs)
    # Trunk branch end

    # Soft mask branch begin
    # Related with r parameter of paper, r=1
    Mpool1 = mx.sym.Pooling(data=RB_first, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max', name='%s_mpool1' %(name)) 
    SM_RB_1 = Residual_unit(data=Mpool1, num_filter=num_filter, stride=stride, dim_match=dim_match, name='%s_sm_rb_1' %(name), bottle_neck=bottle_neck, **kwargs)
    # Skip 1 connection with residual block
    SK_CN_RB_1 = Residual_unit(data=SM_RB_1, num_filter=num_filter, stride=stride, dim_match=dim_match, name='%s_sk_cn_rb_1' %(name), bottle_neck=bottle_neck, **kwargs)
    Mpool2 = mx.sym.Pooling(data=SM_RB_1, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max', name='%s_mpool2' %(name))
    SM_RB_2 = Residual_unit(data=Mpool2, num_filter=num_filter, stride=stride, dim_match=dim_match, name='%s_sm_rb_2' %(name), bottle_neck=bottle_neck, **kwargs)
    # Skip 2 connection with residual block
    SK_CN_RB_2 = Residual_unit(data=SM_RB_2, num_filter=num_filter, stride=stride, dim_match=dim_match, name='%s_sk_cn_rb_2' %(name), bottle_neck=bottle_neck, **kwargs)
    Mpool3 = mx.sym.Pooling(data=SM_RB_2, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max', name='%s_mpool3' %(name))
    # Related with 2r parameter of paper, 2r=2
    SM_RB_3_1 = Residual_unit(data=Mpool3, num_filter=num_filter, stride=stride, dim_match=dim_match, name='%s_sm_rb_3_1' %(name), bottle_neck=bottle_neck, **kwargs)
    SM_RB_3_2 = Residual_unit(data=SM_RB_3_1, num_filter=num_filter, stride=stride, dim_match=dim_match, name='%s_sm_rb_3_2' %(name), bottle_neck=bottle_neck, **kwargs)
    # Summation with soft mask residual block 2
    SM_UP_3 = mx.symbol.contrib.BilinearResize2D(data=SM_RB_3_2, height=size3, width=size3, name='%s_sm_up_3' %(name)) + SM_RB_2
    # Summation with Skip 2 connection
    SM_SUM_CN_2 = SM_UP_3 + SK_CN_RB_2
    SM_RB_4 = Residual_unit(data=SM_SUM_CN_2, num_filter=num_filter, stride=stride, dim_match=dim_match, name='%s_sm_rb_4' %(name), bottle_neck=bottle_neck, **kwargs)
    # Summation with soft mask residual block 1
    SM_UP_2 = mx.symbol.contrib.BilinearResize2D(data=SM_RB_4, height=size2, width=size2, name='%s_sm_up_2' %(name)) + SM_RB_1
    # Summation with Skip 1 connection
    SM_SUM_CN_1 = SM_UP_2 + SK_CN_RB_1
    SM_RB_5 = Residual_unit(data=SM_SUM_CN_1, num_filter=num_filter, stride=stride, dim_match=dim_match, name='%s_sm_rb_5' %(name), bottle_neck=bottle_neck, **kwargs)
    # Summation with trunk branch
    SM_UP_1 = mx.symbol.contrib.BilinearResize2D(data=SM_RB_5, height=size1, width=size1, name='%s_sm_up_1' %(name)) + Trunk_RB_2
    SM_c1x1_a = BN_Act_Conv(SM_UP_1, num_filter=num_filter, kernel=(1, 1), stride=stride, pad=(0, 0), num_group=1, 
                                         momentum=bn_mom, workspace=workspace, name='%s_sm_c1x1_a' %(name))
    SM_c1x1_b = BN_Act_Conv(SM_c1x1_a, num_filter=num_filter, kernel=(1, 1), stride=stride, pad=(0, 0), num_group=1, 
                                         momentum=bn_mom, workspace=workspace, name='%s_sm_c1x1_b' %(name))
    SM_sigmoid = mx.symbol.Activation(data=SM_c1x1_b, act_type='sigmoid', name='%s_sm_sigmoid' %(name))
    # Soft mask branch end

    # Attention summation
    SM_Attent = Trunk_RB_2 + Trunk_RB_2 * SM_sigmoid
    RB_last = Residual_unit(data=SM_Attent, num_filter=num_filter, stride=stride, dim_match=dim_match, name='%s_RB_last' %(name), bottle_neck=bottle_neck, **kwargs)
    return RB_last

# AttentionModule_stage2: input size is 28x28
def AttentionModule_stage2(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs):
    size1 = kwargs.get('size1', 28)
    size2 = kwargs.get('size2', 14)
    bn_mom = kwargs.get('bn_mom', 0.9)
    workspace = kwargs.get('workspace', 256)
    eps = kwargs.get('eps', 2e-5)
    version_se = kwargs.get('version_se', 1)
    act_type = kwargs.get('version_act', 'prelu')
    ibn = kwargs.get('ibn', False)
    memonger = kwargs.get('memonger', False)

    # Related with p parameter of paper, p=1
    RB_first = Residual_unit(data=data, num_filter=num_filter, stride=stride, dim_match=dim_match, name='%s_rb_first' %(name), bottle_neck=bottle_neck, **kwargs)
    # Trunk branch begin
    # Related with t parameter of papar, t=2
    Trunk_RB_1 = Residual_unit(data=RB_first, num_filter=num_filter, stride=stride, dim_match=dim_match, name='%s_tk_rb_1' %(name), bottle_neck=bottle_neck, **kwargs)
    Trunk_RB_2 = Residual_unit(data=Trunk_RB_1, num_filter=num_filter, stride=stride, dim_match=dim_match, name='%s_tk_rb_2' %(name), bottle_neck=bottle_neck, **kwargs)
    # Trunk branch end

    # Soft mask branch begin
    # Related with r parameter of paper, r=1
    Mpool1 = mx.sym.Pooling(data=RB_first, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max', name='%s_mpool1' %(name)) 
    SM_RB_1 = Residual_unit(data=Mpool1, num_filter=num_filter, stride=stride, dim_match=dim_match, name='%s_sm_rb_1' %(name), bottle_neck=bottle_neck, **kwargs)
    # Skip 1 connection with residual block
    SK_CN_RB_1 = Residual_unit(data=SM_RB_1, num_filter=num_filter, stride=stride, dim_match=dim_match, name='%s_sk_cn_rb_1' %(name), bottle_neck=bottle_neck, **kwargs)
    Mpool2 = mx.sym.Pooling(data=SM_RB_1, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max', name='%s_mpool2' %(name))
    # Related with 2r parameter of paper, 2r=2
    SM_RB_2_1 = Residual_unit(data=Mpool2, num_filter=num_filter, stride=stride, dim_match=dim_match, name='%s_sm_rb_2_1' %(name), bottle_neck=bottle_neck, **kwargs)
    SM_RB_2_2 = Residual_unit(data=SM_RB_2_1, num_filter=num_filter, stride=stride, dim_match=dim_match, name='%s_sm_rb_2_2' %(name), bottle_neck=bottle_neck, **kwargs)
    # Summation with soft mask residual block 1
    SM_UP_2 = mx.symbol.contrib.BilinearResize2D(data=SM_RB_2_2, height=size2, width=size2, name='%s_sm_up_2' %(name)) + SM_RB_1
    # Summation with Skip 1 connection
    SM_SUM_CN_1 = SM_UP_2 + SK_CN_RB_1
    SM_RB_3 = Residual_unit(data=SM_SUM_CN_1, num_filter=num_filter, stride=stride, dim_match=dim_match, name='%s_sm_rb_3' %(name), bottle_neck=bottle_neck, **kwargs)
    # Summation with trunk branch
    SM_UP_1 = mx.symbol.contrib.BilinearResize2D(data=SM_RB_3, height=size1, width=size1, name='%s_sm_up_1' %(name)) + Trunk_RB_2
    SM_c1x1_a = BN_Act_Conv(SM_UP_1, num_filter=num_filter, kernel=(1, 1), stride=stride, pad=(0, 0), num_group=1, 
                                         momentum=bn_mom, workspace=workspace, name='%s_sm_c1x1_a' %(name))
    SM_c1x1_b = BN_Act_Conv(SM_c1x1_a, num_filter=num_filter, kernel=(1, 1), stride=stride, pad=(0, 0), num_group=1, 
                                         momentum=bn_mom, workspace=workspace, name='%s_sm_c1x1_b' %(name))
    SM_sigmoid = mx.symbol.Activation(data=SM_c1x1_b, act_type='sigmoid', name='%s_sm_sigmoid' %(name))
    # Soft mask branch end

    # Attention summation
    SM_Attent = Trunk_RB_2 + Trunk_RB_2 * SM_sigmoid
    RB_last = Residual_unit(data=SM_Attent, num_filter=num_filter, stride=stride, dim_match=dim_match, name='%s_rb_last' %(name), bottle_neck=bottle_neck, **kwargs)
    return RB_last

# AttentionModule_stage3: input size is 14x14
def AttentionModule_stage3(data, num_filter, stride, dim_match, name, bottle_neck, **kwargs):
    size1 = kwargs.get('size1', 14)
    bn_mom = kwargs.get('bn_mom', 0.9)
    workspace = kwargs.get('workspace', 256)
    eps = kwargs.get('eps', 2e-5)
    version_se = kwargs.get('version_se', 1)
    act_type = kwargs.get('version_act', 'prelu')
    ibn = kwargs.get('ibn', False)
    memonger = kwargs.get('memonger', False)

    # Related with p parameter of paper, p=1
    RB_first = Residual_unit(data=data, num_filter=num_filter, stride=stride, dim_match=dim_match, name='%s_rb_first' %(name), bottle_neck=bottle_neck, **kwargs)
    # Trunk branch begin
    # Related with t parameter of papar, t=2
    Trunk_RB_1 = Residual_unit(data=RB_first, num_filter=num_filter, stride=stride, dim_match=dim_match, name='%s_tk_rb_1' %(name), bottle_neck=bottle_neck, **kwargs)
    Trunk_RB_2 = Residual_unit(data=Trunk_RB_1, num_filter=num_filter, stride=stride, dim_match=dim_match, name='%s_tk_rb_2' %(name), bottle_neck=bottle_neck, **kwargs)
    # Trunk branch end

    # Soft mask branch begin
    # Related with r parameter of paper, r=1
    Mpool1 = mx.sym.Pooling(data=RB_first, kernel=(3, 3), stride=(2, 2), pad=(1, 1), pool_type='max', name='%s_mpool1' %(name))
    # Related with 2r parameter of paper, 2r=2
    SM_RB_1_1 = Residual_unit(data=Mpool1, num_filter=num_filter, stride=stride, dim_match=dim_match, name='%s_sm_rb_1_1' %(name), bottle_neck=bottle_neck, **kwargs)
    SM_RB_1_2 = Residual_unit(data=SM_RB_1_1, num_filter=num_filter, stride=stride, dim_match=dim_match, name='%s_sm_rb_1_2' %(name), bottle_neck=bottle_neck, **kwargs)
    # Summation with trunk branch
    SM_UP_1 = mx.symbol.contrib.BilinearResize2D(data=SM_RB_1_2, height=size1, width=size1, name='%s_sm_up_1' %(name)) + Trunk_RB_2
    SM_c1x1_a = BN_Act_Conv(SM_UP_1, num_filter=num_filter, kernel=(1, 1), stride=stride, pad=(0, 0), num_group=1, 
                                         momentum=bn_mom, workspace=workspace, name='%s_sm_c1x1_a' %(name))
    SM_c1x1_b = BN_Act_Conv(SM_c1x1_a, num_filter=num_filter, kernel=(1, 1), stride=stride, pad=(0, 0), num_group=1, 
                                         momentum=bn_mom, workspace=workspace, name='%s_sm_c1x1_b' %(name))
    SM_sigmoid = mx.symbol.Activation(data=SM_c1x1_b, act_type='sigmoid', name='%s_sm_sigmoid' %(name))
    # Soft mask branch end

    # Attention summation
    SM_Attent = Trunk_RB_2 + Trunk_RB_2 * SM_sigmoid
    RB_last = Residual_unit(data=SM_Attent, num_filter=num_filter, stride=stride, dim_match=dim_match, name='%s_rb_last' %(name), bottle_neck=bottle_neck, **kwargs)
    return RB_last

def get_symbol(num_classes, num_layers, **kwargs):
	# Residual Attention Network (RAN) architecture
    global bn_mom
    workspace = kwargs.get('workspace', 256)
    eps = kwargs.get('eps', 2e-5)
    bn_mom = kwargs.get('bn_mom', 0.9)
    input_shape = kwargs.get('input_shape', None)
    version_se = kwargs.get('version_se', 1)
    version_output = kwargs.get('version_output', 'E')
    fc_type = version_output

    filter_list = [64, 256, 512, 1024, 2048]
    bottle_neck = True
    if num_layers == 56:
        AM_units = [1, 1, 1]
    elif num_layers == 92:
        AM_units = [1, 2, 3]
    else:
        raise ValueError("no experiments done on num_layers {}, you can do it yourself".format(num_layers))

    data = mx.sym.Variable(name='data', shape=input_shape)
    data = data-127.5
    data = data*0.0078125

    # We refer the input setting of this paper II:
    # Jiankang Deng, Jia Guo and Stefanos Zafeiriou, "ArcFace: Additive Angular Margin Loss for Deep Face Recognition", arXiv:1801.o7698v1
    # The size of input faces is only 112x112 not 224x224
    conv1 = Conv(data, filter_list[0], kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=1, workspace=workspace, name='RA_conv1')
    conv1 = BN_Act(conv1, momentum=bn_mom, name='RA_conv1_bn1')
    conv1._set_attr(mirror_stage='True')

    RAN_RU_1 = Residual_unit(data=conv1, num_filter=filter_list[1], stride=(2, 2), dim_match=False, name='RAN_R_unit1', bottle_neck=bottle_neck, **kwargs)
    RAN_RU_1._set_attr(mirror_stage='True')
    for j in range(AM_units[0]):
        RAN_RU_1 = AttentionModule_stage1(data=RAN_RU_1, num_filter=filter_list[1], stride=(1, 1), dim_match=True, name='RAN_AM_s1_unit%d' % (j+1), bottle_neck=bottle_neck, **kwargs)
        RAN_RU_1._set_attr(mirror_stage='True')
    RAN_RU_2 = Residual_unit(data=RAN_RU_1, num_filter=filter_list[2], stride=(2, 2), dim_match=False, name='RAN_R_unit2', bottle_neck=bottle_neck, **kwargs)
    RAN_RU_2._set_attr(mirror_stage='True')
    for j in range(AM_units[1]):
        RAN_RU_2 = AttentionModule_stage2(data=RAN_RU_2, num_filter=filter_list[2], stride=(1, 1), dim_match=True, name='RAN_AM_s2_unit%d' % (j+1), bottle_neck=bottle_neck, **kwargs)
        RAN_RU_2._set_attr(mirror_stage='True')
    RAN_RU_3 = Residual_unit(data=RAN_RU_2, num_filter=filter_list[3], stride=(2, 2), dim_match=False, name='RAN_R_unit3', bottle_neck=bottle_neck, **kwargs)
    RAN_RU_3._set_attr(mirror_stage='True')
    for j in range(AM_units[2]):
        RAN_RU_3 = AttentionModule_stage3(data=RAN_RU_3, num_filter=filter_list[3], stride=(1, 1), dim_match=True, name='RAN_AM_s3_unit%d' % (j+1), bottle_neck=bottle_neck, **kwargs)
        RAN_RU_3._set_attr(mirror_stage='True')
    RAN_RU_4 = Residual_unit(data=RAN_RU_3, num_filter=filter_list[4], stride=(2, 2), dim_match=False, name='RAN_R_unit4_1', bottle_neck=bottle_neck, **kwargs)
    RAN_RU_4._set_attr(mirror_stage='True')
    for j in range(2):
        RAN_RU_4 = Residual_unit(data=RAN_RU_4, num_filter=filter_list[4], stride=(1, 1), dim_match=True, name='RAN_R_unit4_%d' % (j+2), bottle_neck=bottle_neck, **kwargs)
        RAN_RU_4._set_attr(mirror_stage='True')
    fc1 = symbol_utils.get_fc1(RAN_RU_4, num_classes, fc_type)
    fc1._set_attr(mirror_stage='True')
    return fc1
