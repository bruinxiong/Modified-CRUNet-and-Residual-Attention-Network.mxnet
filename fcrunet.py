import mxnet as mx
import symbol_utils
# Coded by Lin Xiong on Jun-6, 2018
# Modified from https://github.com/cypw/CRU_Net, 
# More detailed information can be found in the following paper I:
# Chen Yunpeng, Jin Xiaojie, Kang Bingyi, Feng Jiashi and Yan Shuicheng, "Sharing Residual Units Through Collective Tensor Factorization in Deep Neural Networks", arXiv:1703.02180v2, IJCAI-ECAI, Stockolm, Sweden, July 13-19, 2018
# We also refer the input setting of this paper II:
# Jiankang Deng, Jia Guo and Stefanos Zafeiriou, "ArcFace: Additive Angular Margin Loss for Deep Face Recognition", arXiv:1801.o7698v1
# The size of input faces is only 112x112 not 224x224

bn_mom = 0.9
#bn_mom = 0.9997

def BN(data, momentum=bn_mom, fix_gamma=False, name=None, suffix=''):
    bn = mx.sym.BatchNorm(data=data, name='%s%s_batchnorm' %(name, suffix), fix_gamma=fix_gamma, momentum=momentum, cudnn_off=True)
    # bn = mx.sym.BatchNorm(data=data, name='%s%s_batchnorm' %(name, suffix), fix_gamma=fix_gamma, momentum=momentum)
    return bn

def Act(data, act_type='prelu', name=None):
    body = mx.sym.LeakyReLU(data = data, act_type=act_type, name = '%s_%s' %(name, act_type))
    #body = mx.sym.Activation(data=data, act_type='relu', name=name)
    return body

def Conv(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, name=None, w=None, b=None, suffix=''):
    if w is None:
        conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, num_group=num_group, stride=stride, pad=pad, no_bias=True, name='%s%s_conv2d' %(name, suffix))
    else:
        if b is None:
            conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, num_group=num_group, stride=stride, pad=pad, no_bias=True, weight=w, name='%s%s_conv2d' %(name, suffix))
        else:
            conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, num_group=num_group, stride=stride, pad=pad, weight=w, bias=b, name='%s%s_conv2d' %(name, suffix))
    return conv

def BN_Act(data, momentum=bn_mom, name=None, suffix=''):
    bn = BN(data, momentum=momentum, fix_gamma=False, name=name, suffix=suffix)
    # bn = mx.sym.BatchNorm(data=data, name='%s%s_batchnorm' %(name, suffix), fix_gamma=False, momentum=momentum, cudnn_off=True)
    # bn = mx.sym.BatchNorm(data=data, name='%s%s_batchnorm' %(name, suffix), fix_gamma=False, momentum=momentum)
    bn_act = Act(bn, act_type='prelu', name=name)
    return bn_act

def BN_Conv(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, momentum=bn_mom, name=None, w=None, b=None, suffix=''):
    bn = BN(data, momentum=momentum, fix_gamma=False, name=name, suffix=suffix)
    # bn = mx.sym.BatchNorm(data=data, name='%s%s_batchnorm' %(name, suffix), fix_gamma=False, momentum=momentum, cudnn_off=True)
    # bn = mx.sym.BatchNorm(data=data, name='%s%s_batchnorm' %(name, suffix), fix_gamma=False, momentum=momentum)
    bn_conv = Conv(bn, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, num_group=num_group, name=name, w=w, b=b, suffix=suffix)
    return bn_conv

def BN_Act_Conv(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, momentum=bn_mom, name=None, w=None, b=None, suffix=''):
    bn = BN(data, momentum=momentum, fix_gamma=False, name=name, suffix=suffix)
    # bn = mx.sym.BatchNorm(data=data, name='%s%s_batchnorm' %(name, suffix), fix_gamma=False, momentum=momentum, cudnn_off=True)
    # bn = mx.sym.BatchNorm(data=data, name='%s%s_batchnorm' %(name, suffix), fix_gamma=False, momentum=momentum)
    bn_act = Act(bn, act_type='prelu', name=name)
    bn_act_conv = Conv(bn_act, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, num_group=num_group, name=name, w=w, b=b, suffix=suffix)
    # if w is None:
    #     # conv = mx.sym.Convolution(data=act, num_filter=num_filter, kernel=kernel, num_group=num_group, stride=stride, pad=pad, no_bias=True, name='%s%s_conv2d' %(name, suffix))
    #     conv = Conv(act, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, num_group=num_group, no_bias=True, name='%s%s_conv2d' %(name, suffix))
    # else:
    #     if b is None:
    #         # conv = mx.sym.Convolution(data=act, num_filter=num_filter, kernel=kernel, num_group=num_group, stride=stride, pad=pad, no_bias=True, weight=w, name='%s%s_conv2d' %(name, suffix))
    #         conv = Conv(act, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, num_group=num_group, no_bias=True, name='%s%s_conv2d' %(name, suffix), w=w)
    #     else:
    #         # conv = mx.sym.Convolution(data=act, num_filter=num_filter, kernel=kernel, num_group=num_group, stride=stride, pad=pad, weight=w, bias=b, name='%s%s_conv2d' %(name, suffix))
    #         conv = Conv(act, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, num_group=num_group, name='%s%s_conv2d' %(name, suffix), w=w, b=b)
    return bn_act_conv

def Shared_Weights(R, nc, name):
    dim0 = R  # out
    dim1 = nc # in
    dim2 = 1
    dim3 = 1
    weights_dim3 = mx.symbol.Variable(name='%s_%dx%d_bases[dim3]_weight' %(name, dim2, dim3), shape=(dim0, dim1, dim2, dim3))

    dim0 = R
    dim1 = 1
    dim2 = 3
    dim3 = 3
    weights_dim21 = mx.symbol.Variable(name='%s_%dx%d_bases[dim21]_weight' %(name, dim2, dim3), shape=(dim0, dim1, dim2, dim3))

    weights = {'dim3':weights_dim3, 'dim21':weights_dim21}
    return weights

# Standard Residual Units
def Residual_unit(data, num_filter, R, name, momentum, _type='normal', **kwargs):
    memonger = kwargs.get('memonger', False)
    if _type is 'proj':
        key_stride = 1
        has_proj   = True
    if _type is 'down':
        key_stride = 2
        has_proj   = True
    if _type is 'normal':
        key_stride = 1
        has_proj   = False

    if has_proj:
        data_o    = data
        # w_weight  = mx.symbol.Variable(name='%s_c1x1-w_weight' %(name))
        # c1x1_w    = BN_Act_Conv(data, num_filter=num_filter, kernel=(1, 1), stride=(key_stride, key_stride), pad=(0, 0), num_group=1, name='%s_c1x1-w(s/%d)' %(name, key_stride), w=w_weight)
        # c1x1_w    = BN_Act_Conv(data, num_filter=num_filter, kernel=(1, 1), stride=(key_stride, key_stride), pad=(0, 0), num_group=1, momentum=momentum, name='%s_c1x1-w(s/%d)' %(name, key_stride))
        c1x1_w    = Conv(data, num_filter=num_filter, kernel=(1, 1), stride=(key_stride, key_stride), pad=(0, 0), num_group=1, name='%s_c1x1-w(s/%d)' %(name, key_stride), w=None, b=None, suffix='')
        c1x1_w    = BN(c1x1_w, momentum=momentum, fix_gamma=False, name='%s_bn-c1x1-w(s/%d)' %(name, key_stride), suffix='')
    else:
        data_o    = data
        c1x1_w    = data
    if memonger:
        c1x1_w._set_attr(mirror_stage='True')

    c1x1_a = BN_Conv(data_o, num_filter=int(num_filter*0.5), kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, momentum=momentum, name='%s_c1x1-a' %(name))
    # c1x1_a = BN_Act_Conv(data_o, num_filter=int(num_filter*0.5), kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, momentum=momentum, name='%s_c1x1-a' %(name))
    # c3x3_b = BN_Act_Conv(c1x1_a, num_filter=int(num_filter*0.5), kernel=(3, 3), stride=(key_stride, key_stride), pad=(1, 1), num_group=R, momentum=momentum, name='%s_c3x3-b' %(name))
    c3x3_b = BN_Act_Conv(c1x1_a, num_filter=int(num_filter*0.5), kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=R, momentum=momentum, name='%s_c3x3-b' %(name))
    # c1x1_c = BN_Act_Conv(c3x3_b, num_filter=num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, momentum=momentum, name='%s_c1x1-c' %(name))
    c1x1_c = BN_Act_Conv(c3x3_b, num_filter=num_filter, kernel=(1, 1), stride=(key_stride, key_stride), pad=(0, 0), num_group=1, momentum=momentum, name='%s_c1x1-c' %(name))
    c1x1_c = BN(c1x1_c, momentum=momentum, fix_gamma=False, name='%s_bn-c1x1-c' %(name))

    # summ   = mx.symbol.ElementWiseSum(*[c1x1_w, c1x1_c], name='%s_ele-sum' %(name))
    summ   = c1x1_c + c1x1_w
    return summ

# Collective Residual Units
def CR_unit(data, num_filter_in, num_filter_out, R, name, momentum, weights, _type='normal', **kwargs):
    memonger = kwargs.get('memonger', False)
    if _type is 'proj':
        key_stride = 1
        has_proj   = True
    if _type is 'down':
        key_stride = 2
        has_proj   = True
    if _type is 'normal':
        key_stride = 1
        has_proj   = False

    if has_proj:
        # w_weight  = mx.symbol.Variable(name='%s_c1x1-w_weight' %(name))
        # data_o    = BN_Act_Conv(data, num_filter=num_filter_out, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, name='%s_c1x1-w(s/1)' %(name), w=w_weight)
        # data_o    = BN_Act_Conv(data, num_filter=num_filter_out, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, name='%s_c1x1-w(s/1)' %(name))
        data_o    = Conv(data, num_filter=num_filter_out, kernel=(1, 1), pad=(0, 0), num_group=1, name='%s_c1x1-w(s/1)' %(name), w=None, b=None, suffix='')
        data_o    = BN(data_o, momentum=momentum, fix_gamma=False, name='%s_bn-c1x1-w(s/1)' %(name), suffix='')
        if key_stride > 1:
            # c1x1_w    = BN_Act_Conv(data, num_filter=num_filter_out, kernel=(1, 1), stride=(key_stride, key_stride), pad=(0, 0), num_group=1, name='%s_c1x1-w(s/key)' %(name), w=w_weight)
            # c1x1_w    = BN_Act_Conv(data, num_filter=num_filter_out, kernel=(1, 1), stride=(key_stride, key_stride), pad=(0, 0), num_group=1, name='%s_c1x1-w(s/key)' %(name))
            c1x1_w    = Conv(data, num_filter=num_filter_out, kernel=(1, 1), stride=(key_stride, key_stride), pad=(0, 0), num_group=1, name='%s_c1x1-w(s/key)' %(name), w=None)
            c1x1_w    = BN(c1x1_w, momentum=momentum, fix_gamma=False, name='%s_bn-c1x1-w(s/key)' %(name), suffix='')
        else:
            c1x1_w    = data_o
    else:
        data_o    = data
        c1x1_w    = data
    if memonger:
        c1x1_w._set_attr(mirror_stage='True')

    c1x1_a = BN_Act_Conv(data_o, num_filter=num_filter_in, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, momentum=momentum, name='%s_c1x1-a' % (name), w=weights['dim3'])
    # c1x1_a = BN_Act_Conv(data_o, num_filter=num_filter_in, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, momentum=momentum, name='%s_c1x1-a' % (name))
    # c3x3_b = Conv(c1x1_a, num_filter=num_filter_in, kernel=(3, 3), stride=(key_stride, key_stride), pad=(1, 1), num_group=R, momentum=momentum, name='%s_c3x3-b(1)' %(name), w=weights['dim21'])
    # c3x3_b = Conv(c1x1_a, num_filter=num_filter_in, kernel=(3, 3), stride=(key_stride, key_stride), pad=(1, 1), num_group=R, momentum=momentum, name='%s_c3x3-b(1)' %(name))
    c3x3_b = BN_Act_Conv(c1x1_a, num_filter=num_filter_in, kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=R, momentum=momentum, name='%s_c3x3-b(1)' %(name), w=weights['dim21'])
    c1x1_b = BN_Act_Conv(c3x3_b, num_filter=num_filter_in, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, momentum=momentum, name='%s_c1x1-b' %(name))
    # c1x1_c = BN_Act_Conv(c1x1_b, num_filter=num_filter_out, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, momentum=momentum, name='%s_c1x1-c' %(name))
    c1x1_c = BN_Act_Conv(c1x1_b, num_filter=num_filter_out, kernel=(1, 1), stride=(key_stride, key_stride), pad=(0, 0), num_group=1, momentum=momentum, name='%s_c1x1-c' %(name))
    c1x1_c = BN(c1x1_c, momentum=momentum, fix_gamma=False, name='%s_bn-c1x1-c' %(name))

    # summ   = mx.symbol.ElementWiseSum(*[c1x1_w, c1x1_c], name='%s_ele-sum' %(name))
    summ   = c1x1_c + c1x1_w
    return summ


def get_symbol(num_classes, num_layers, **kwargs):
    global bn_mom
    bn_mom = kwargs.get('bn_mom', 0.9)
    input_shape = kwargs.get('input_shape', None)
    # if we let kwargs.get('version_output', 'A'), it is the same as the paper I
    version_output = kwargs.get('version_output', 'E') # it is the same as the paper II
    fc_type = version_output
    R = 32

    filter_list = [64, 256, 512, 1024, 2048]
    if num_layers == 56:
        units = [3, 4, 6, 3]
        k_D   = [320]
    elif num_layers == 116:
        units = [3, 6, 18, 3]
        k_D   = [176]
        # k_D   = [128, 176]
    else:
        raise ValueError("no experiments done on num_layers {}, you can do it yourself".format(num_layers))

    data = mx.sym.Variable(name='data', shape=input_shape)
    data = data-127.5
    data = data*0.0078125

    # We refer the input setting of this paper II:
    # Jiankang Deng, Jia Guo and Stefanos Zafeiriou, "ArcFace: Additive Angular Margin Loss for Deep Face Recognition", arXiv:1801.o7698v1
    # The size of input faces is only 112x112 not 224x224
    conv1 = Conv(data, filter_list[0], kernel=(3, 3), stride=(1, 1), pad=(1, 1), num_group=1, name='conv1')
    conv1 = BN_Act(conv1, momentum=bn_mom, name='conv1-bn1')
    conv1._set_attr(mirror_stage='True')

    conv2_x = Residual_unit(conv1, filter_list[1], R, 'conv2_x_1', bn_mom, 'down', **kwargs)
    conv2_x._set_attr(mirror_stage='True')
    for i in range(2, units[0]+1):
        conv2_x = Residual_unit(conv2_x, filter_list[1], R, 'conv2_x_%d' %(i), bn_mom, 'normal', **kwargs)
        conv2_x._set_attr(mirror_stage='True')

    nc = filter_list[2]
    weights = Shared_Weights(k_D[0]*2, nc, 'conv3_x')
    # weights = None
    conv3_x = CR_unit(conv2_x, k_D[0]*2, filter_list[2], k_D[0]*2, 'conv3_x_1', bn_mom, weights, 'down', **kwargs)
    # conv3_x = CR_unit(conv2_x, k_D[0]*2, filter_list[2], R, 'conv3_x_1', bn_mom, weights, 'down', **kwargs)
    conv3_x._set_attr(mirror_stage='True')
    for i in range(2, units[1]+1):
        conv3_x = CR_unit(conv3_x, k_D[0]*2, filter_list[2], k_D[0]*2, 'conv3_x_%d' %(i), bn_mom, weights, 'normal', **kwargs)
        # conv3_x = CR_unit(conv3_x, k_D[0]*2, filter_list[2], R, 'conv3_x_%d' %(i), bn_mom, weights, 'normal', **kwargs)
        conv3_x._set_attr(mirror_stage='True')

    nc = filter_list[3]
    weights = Shared_Weights(k_D[0]*4, nc, 'conv4_x_1')
    # weights = None
    conv4_x = CR_unit(conv3_x, k_D[0]*4, filter_list[3], k_D[0]*4, 'conv4_x_1', bn_mom, weights, 'down', **kwargs)
    # conv4_x = CR_unit(conv3_x, k_D[0]*4, filter_list[3], R, 'conv4_x_1', bn_mom, weights, 'down', **kwargs)
    conv4_x._set_attr(mirror_stage='True')
    for i in range(2, units[2]+1):
        if (i%6) == 1:
            weights = Shared_Weights(k_D[0]*4, nc, 'conv4_x_%d' %(int(i/6)+1))
            # weights = None
        conv4_x = CR_unit(conv4_x, k_D[0]*4, filter_list[3], k_D[0]*4, 'conv4_x_%d' %(i), bn_mom, weights, 'normal', **kwargs)
        # conv4_x = CR_unit(conv4_x, k_D[0]*4, filter_list[3], R, 'conv4_x_%d' %(i), bn_mom, weights, 'normal', **kwargs)
        conv4_x._set_attr(mirror_stage='True')

    conv5_x = Residual_unit(conv4_x, filter_list[4], R, 'conv5_x_1', bn_mom, 'down', **kwargs)
    conv5_x._set_attr(mirror_stage='True')
    for i in range(2, units[3]+1):
        conv5_x = Residual_unit(conv5_x, filter_list[4], R, 'conv5_x_%d' %(i), bn_mom, 'normal', **kwargs)
        conv5_x._set_attr(mirror_stage='True')

    fc1 = symbol_utils.get_fc1(conv5_x, num_classes, fc_type)
    fc1._set_attr(mirror_stage='True')
    return fc1
