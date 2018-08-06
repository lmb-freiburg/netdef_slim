from netdef_slim.core.register import register_op
import tensorflow as tf
from .blob import _slice
from lmbspecialops import lut_function1d
import numpy as np
import netdef_slim as nd

nothing = None

# ----------------------------------------------------------------------
def _threshold(tensor, thresh):
    condition = tf.less(tensor, thresh)
    return tf.where(condition, _const_like(tensor, 0.0), _const_like(tensor, 1.0))

register_op('threshold', _threshold)

# ----------------------------------------------------------------------
_scale_conv_n=0
def _scale(tensor, factor):
    '''
    Factor can be a scalar or a list. In case of a list
    each channel is scaled by a different factor.
    '''
    global _scale_conv_n
    if type(factor) is list or type(factor) is tuple:
        _scale_conv_n += 1
        kernel = tf.constant(factor)
        kernel = tf.reshape(kernel, (1,1,len(factor),1))
        return tf.nn.depthwise_conv2d(tensor,
                            filter=kernel,
                            strides=[1,1,1,1],
                            padding='VALID',
                            data_format='NCHW')

    else:
        return tf.multiply(tensor, factor)

register_op('scale', _scale)

# ----------------------------------------------------------------------
def _zeros(num, channels, height, width, include_phase=None):
    return tf.zeros((num, channels, height, width))

register_op('zeros', _zeros)

# ----------------------------------------------------------------------
def _zeros_like(other, include_phase=None):
    return tf.zeros_like(other)

register_op('zeros_like', _zeros_like)

# ----------------------------------------------------------------------
def _ones(num, channels, height, width, include_phase=None):
    return tf.ones((num, channels, height, width))

register_op('ones', _ones)

# ----------------------------------------------------------------------
def _ones_like(other, include_phase=None):
    return tf.ones_like(other)

register_op('ones_like', _ones_like)

# ----------------------------------------------------------------------
def _constant(num, channels, height, width, value):
    return tf.fill((num, channels, height, width), value)

register_op('constant', _constant)

# ----------------------------------------------------------------------
def _const_like(other, value):
    return tf.fill(other.get_shape(), value)

register_op('const_like', _const_like)

# ----------------------------------------------------------------------
def _abs(A):
    return tf.abs(A)

register_op('abs', _abs)

# ----------------------------------------------------------------------
def _add(A, B, coeffA=None, coeffB=None):
    if coeffA is None:
        coeffA = tf.constant(1, dtype=A.dtype)
    if coeffB is None:
        coeffB = tf.constant(1, dtype=B.dtype)
    #return tf.add(tf.multiply(tf.to_float(A), coeffA), tf.multiply(tf.to_float(B), coeffB))
    return tf.add(tf.multiply(A, coeffA), tf.multiply(B, coeffB))

register_op('add', _add)

# ----------------------------------------------------------------------
def _sub(A, B):
    return tf.subtract(A, B)

register_op('sub', _sub)

# ----------------------------------------------------------------------
def _mul(A, B):
    return tf.multiply(A, B)

register_op('mul', _mul)

# ----------------------------------------------------------------------
def _const_mul(coeff, A):
    return tf.scalar_mul(coeff, A)

register_op('const_mul', _const_mul)

# ----------------------------------------------------------------------
def _channel_norm(blob):
    out = tf.sqrt(tf.reduce_sum(tf.square(blob), axis=1))
    return tf.expand_dims(out, axis = 1)


register_op('channel_norm', _channel_norm)
# ----------------------------------------------------------------------
def _sqrt(x):
    return tf.pow(x, 0.5)

register_op('sqrt', _sqrt)

# ----------------------------------------------------------------------
def _sqr(x):
    return tf.pow(x, 2)

register_op('sqr', _sqr)

# ----------------------------------------------------------------------
def _exp(X, scale=1.0):
    return tf.exp(_mul(X, scale))

register_op('exp', _exp)

# ----------------------------------------------------------------------
def _log(x):
    return tf.log(x)

register_op('log', _log)

# ----------------------------------------------------------------------
def _inv(x):
    return tf.pow(x, -1)

register_op('inv', _inv)

# ----------------------------------------------------------------------
def _flip_sign(x):
    return _scale(x, -1)

register_op('flip_sign', _flip_sign)

# ----------------------------------------------------------------------
def _spatial_epe(tensor1, tensor2):
    diff = tensor1 - tensor2
    return _channel_norm(diff)

register_op('spatial_epe', _spatial_epe)

# ----------------------------------------------------------------------
def _softmax(X, axis=1):
    return tf.nn.softmax(X, dim=axis)

register_op('softmax', _softmax)

# ----------------------------------------------------------------------
def _sigmoid(X):
    return tf.sigmoid(X)

register_op('sigmoid', _sigmoid)

# ----------------------------------------------------------------------
def _add_eps(X, eps=1e-2 / 2.0):
    return _add(X, _const_like(X, eps))

register_op('add_eps', _add_eps)

# ----------------------------------------------------------------------
def _adjusted_sigmoid(X, min, max):
    tf.add_to_collection('log_scale_bound', max)

    const = lambda z: _const_like(X, z)

    range = max - min
    x_scaled = _mul(X, const(4.0 / range))

    sig = _sigmoid(x_scaled)

    sig_scaled = _mul(sig, const(range))

    if min != 0:
        sig_scaled_shifted = _add(sig_scaled, const(min))
    else:
        sig_scaled_shifted = sig_scaled

    return sig_scaled_shifted

register_op('adjusted_sigmoid', _adjusted_sigmoid)

# ----------------------------------------------------------------------
def _adjusted_sigmoid_bounded_multi(X, boundaries=[], values=[]):

    global_step = nd.scope.global_step()
    if global_step is None:
        global_step = nd.evo_manager.get_last_present_state().iter()
    step = tf.to_float(global_step)
    log_scale_bound = tf.train.piecewise_constant(step, boundaries, values)
    bounded_log_scale = _adjusted_sigmoid(X, -log_scale_bound, log_scale_bound)

    return bounded_log_scale

register_op('adjusted_sigmoid_bounded_multi', _adjusted_sigmoid_bounded_multi)

# ----------------------------------------------------------------------
def _adjusted_sigmoid_bounded_sigmoid(X, max_bound=5.0):

    #5/(1+exp(-7/100x+3)) : where 100 is the max_iter and 5 is max_bound
    global_step = nd.scope.global_step()
    if global_step is None:
        global_step = nd.evo_manager.get_last_present_state().iter()
    step = tf.to_float(global_step)
    if nd.evo_manager.current_evolution() is None:
        max_iter = nd.evo_manager.last_trained_evolution().max_iter()
    else:
        max_iter = nd.evo_manager.current_evolution().max_iter()
    step_scaled = _add(_mul(step, _const_like(step, (7.0/max_iter))), _const_like(step, -3.0))
    step_sigmoid = _sigmoid(step_scaled)
    log_scale_bound = _mul(step_sigmoid, _const_like(step_sigmoid, max_bound))
    bounded_log_scale = _adjusted_sigmoid(X, -log_scale_bound, log_scale_bound)

    return bounded_log_scale

register_op('adjusted_sigmoid_bounded_sigmoid', _adjusted_sigmoid_bounded_sigmoid)

# ----------------------------------------------------------------------
def _spatial_iul_nll(mean, scale_param_log, gt):
    tf.add_to_collection('pred', mean)
    tf.add_to_collection('gt', gt)
    tf.add_to_collection('scale_param_log', scale_param_log)

    eps = 1e-2 / 2.0
    diff = _sub(gt, mean)
    diff2 = _sqr(diff)
    diff2_x, diff2_y = _slice(diff2, 1)
    diff3 = _add(diff2_x, _const_like(diff2_x, eps), 1.0, 1.0)
    diff4 = _add(diff2_y, _const_like(diff2_y, eps), 1.0, 1.0)
    diff5 = _sqrt(diff3)
    diff6 = _sqrt(diff4)

    sp_sx, sp_sy = _slice(scale_param_log, 1)

    sxe = _exp(sp_sx, scale=-1.0)
    sye = _exp(sp_sy, scale=-1.0)

    sxsy = _add(sp_sx, sp_sy)

    e1 = _mul(diff5, sxe)
    e2 = _mul(diff6, sye)
    e = _add(e1, e2)

    total = _add(sxsy, e)

    return total

register_op('spatial_iul_nll', _spatial_iul_nll)

# ----------------------------------------------------------------------
def _spatial_gmm_nll(mean, scale_param_log, mixture_weight, gt):
    tf.add_to_collection('pred', mean)
    tf.add_to_collection('gt', gt)
    tf.add_to_collection('scale_param_log', scale_param_log)
    tf.add_to_collection('mixture_weight', mixture_weight)

    eps = 1e-5 / 2.0
    diff = _sub(gt, mean)
    diff2 = _sqr(diff)
    diff2_x, diff2_y = _slice(diff2, 1)
    sp_sx, sp_sy = _slice(scale_param_log, 1)
    sxe = _exp(sp_sx, scale=1.0)
    sye = _exp(sp_sy, scale=1.0)
    sxe_sq_inv = _inv(_add_eps(_sqr(sxe), eps))
    sye_sq_inv = _inv(_add_eps(_sqr(sye), eps))
    c1 = _mul(diff2_x, sxe_sq_inv)
    c2 = _mul(diff2_y, sye_sq_inv)
    c = _add(c1, c2)
    c_exp = _exp(c, scale=-0.5)
    sxsy = _mul(sxe, sye)
    sxsy_scaled = _inv(_add_eps(_mul(sxsy, _const_like(sxsy, 2 * 3.14)), eps))
    final = _mul(c_exp, sxsy_scaled)
    final_weighted = _mul(final, mixture_weight)

    return final_weighted

register_op('spatial_gmm_nll', _spatial_gmm_nll)


# ----------------------------------------------------------------------
def _spatial_iul_nll_tradeoff(mean, scale_param_log, gt, log_limit_min=-5.0, log_limit_max=5.0):
    tf.add_to_collection('pred', mean)
    tf.add_to_collection('gt', gt)
    tf.add_to_collection('scale_param_log', scale_param_log)

    eps = 1e-2 / 2.0

    diff = _sub(gt, mean)
    diff2 = _sqr(diff)
    diff2_x, diff2_y = _slice(diff2, 1)
    diff3 = _add(diff2_x, _const_like(diff2_x, eps), 1.0, 1.0)
    diff4 = _add(diff2_y, _const_like(diff2_y, eps), 1.0, 1.0)
    diff5 = _sqrt(diff3)
    diff6 = _sqrt(diff4)

    sp_sx, sp_sy = _slice(scale_param_log, 1)
    # sp_sx = _adjusted_sigmoid(sx, min=log_limit_min, max=log_limit_max)
    # sp_sy = _adjusted_sigmoid(sy, min=log_limit_min, max=log_limit_max)
    sxe = _exp(sp_sx, scale=-1.0)
    sye = _exp(sp_sy, scale=-1.0)

    sxsy = _add(sp_sx, sp_sy)

    e1 = _mul(diff5, sxe)
    e2 = _mul(diff6, sye)
    e = _add(e1, e2)

    epe = _spatialEPE(gt, mean)
    tradeoff = tf.get_variable('Variable', initializer=0.0, trainable=False)
    tf.add_to_collection('iul_tradeoff', tradeoff)
    total = _add(sxsy, e) * tradeoff + epe * (1.0 - tradeoff)

    return total


register_op('spatial_iul_nll_tradeoff', _spatial_iul_nll_tradeoff)

# ----------------------------------------------------------------------
def _bessel_k0(x, max=5000, step=0.01):
    from scipy.special import k0
    k0_values = list(k0(np.arange(1e-10, max, step)))
    k0 = lut_function1d(x, values=k0_values, start=1e-10, step=step, clamp_left=True, clamp_right=True)
    k0 = tf.clip_by_value(k0, 0.0, 500.0)
    return k0

# register_op('bessel_k0', _bessel_k0)

## TODO reduce number of operations, look at the last line in original caffe: ... net.constLike(log_d_2, 10*2^(level-1)))
# ----------------------------------------------------------------------
def _spatial_gbl_nll(mean, l1, l2, l3, gt, log_limit_min=-1.0, log_limit_max=1.0):
    tf.add_to_collection('pred', mean)

    eps1 = 1e-4
    eps2 = 1e-30
    eps3 = 1e-6
    eps4 = 1e-4

    diff = gt-mean
    diff_x, diff_y = _slice(diff, 1)
    diff2_x = tf.square(diff_x)
    diff2_y = tf.square(diff_y)
    diff2_xy = diff_x * diff_y

    # log_limit_min = -1.0
    # log_limit_max = 1.0
    l1 = _adjusted_sigmoid(l1, 0, log_limit_max)
    l3 = _adjusted_sigmoid(l3, 0, log_limit_max)
    l2 = _adjusted_sigmoid(l2, log_limit_min, log_limit_max)
    l1 = tensor_stats(l1, 'l1')
    l2 = tensor_stats(l2, 'l2')
    l3 = tensor_stats(l3, 'l3')
    tf.add_to_collection('gbl_l1', l1)
    tf.add_to_collection('gbl_l2', l2)
    tf.add_to_collection('gbl_l3', l3)

    bp1 = tf.square(l1) + eps3
    bp2 = l1*l2
    bp3 = tf.square(l2) + tf.square(l3) + eps3

    d = tf.square(l1)*tf.square(l3) + tf.square(eps3)
    tf.add_to_collection('gbl_d', d)
    d = tensor_stats(d, 'gbl_d')
    log_d_2 = tf.log(d)#/2.0
    log_d_2 = tensor_stats(log_d_2, 'log_d_2')
    bp2_2 = -2.0*bp2

    xx = bp3 * diff2_x
    yy = bp1 * diff2_y
    xy = bp2_2 * diff2_xy

    alpha = xx+xy+yy
    #alpha = tensor_stats(alpha, 'alpha')
    alpha_2 = alpha *2.0
    alpha_2_div_d = alpha_2/(d+eps4)
    alpha_sqrt = tf.sqrt(alpha_2_div_d+eps1)
    alpha_sqrt = tensor_stats(alpha_sqrt, 'alpha_sqrt')

    K0 = _bessel_k0(alpha_sqrt, max=100, step=0.005)
    tf.add_to_collection('gbl_K0', K0)
    K0 = tensor_stats(K0, 'K0')
    K0_log = tf.log(K0+eps2)

    l = log_d_2-K0_log
    l = tensor_stats(l, 'l')
    return l

register_op('spatial_gbl_nll', _spatial_gbl_nll)

def _spatial_gbl_nll_2(mean, l1, l2, l3, gt, log_limit_min=-1.0, log_limit_max=1.0):
    print('###### USING GBL ALT FORM ######')
    tf.add_to_collection('pred', mean)
    tf.add_to_collection('gbl_l1', l1)
    tf.add_to_collection('gbl_l2', l2)
    tf.add_to_collection('gbl_l3', l3)

    eps1 = 1e-4
    eps2 = 1e-30
    eps3 = 1e-6
    eps4 = 1e-4

    s1 = tf.sigmoid(l1)+eps3
    s2 = tf.sigmoid(l2)+eps3
    r = tf.clip_by_value(tf.tanh(l3), 0.0, 0.9)

    diff = gt - mean
    diff_x, diff_y = _slice(diff, 1)

    xx = tf.square(diff_x)/(tf.square(s1))
    yy = tf.square(diff_y)/(tf.square(s2))
    xy = r*diff_x*diff_y/(s1*s2)

    xx = tensor_stats(xx, 'xx')
    xy = tensor_stats(xy, 'xy')
    yy = tensor_stats(yy, 'yy')

    denominator = 1-tf.square(r)
    denominator = tensor_stats(denominator, 'denominator')

    alpha = 2*(xx-2*xy+yy)/(denominator+eps3)+eps3
    alpha = tensor_stats(alpha, 'alpha')
    alpha_sqrt = tf.sqrt(alpha+eps3)
    alpha_sqrt = tensor_stats(alpha_sqrt, 'alpha_sqrt')
    K0 = _bessel_k0(alpha_sqrt+1e-6, max=100, step=0.005)+eps3
    tf.add_to_collection('gbl_K0', K0)
    K0 = tensor_stats(K0, 'K0')

    pi = 3.141592653589793
    likelihood = K0*(1/(pi*s1*s2*tf.sqrt(1-tf.square(r)))+eps3)
    likelihood = tensor_stats(likelihood, 'likelihood')

    d = (1-tf.square(r))*tf.square(s1*s2)
    d = tensor_stats(d, 'gbl_d')
    tf.add_to_collection('gbl_d', d)

    l = d-likelihood
    l = tensor_stats(l, 'l')

    return l

#register_op('spatial_gbl_nll', _spatial_gbl_nll_2)

# ----------------------------------------------------------------------
def _derivative(blob, direction, order=1, extent=2):
    if order!=1:
        raise NotImplementedError
    if extent!=2:
        raise NotImplementedError
    if direction == 'x':
        blob = tf.pad(blob, [[0, 0], [0, 0], [0, 0], [0, 1]])
        return tf.abs(blob[:, :, :, 1:] - blob[:, :, :, :-1])
    elif direction == 'y':
        blob = tf.pad(blob, [[0, 0], [0, 0], [0, 1], [0, 0]])
        return tf.abs(blob[:, :, 1:, :] - blob[:, :, :-1, :])

register_op('derivative', _derivative)

# ----------------------------------------------------------------------
def _arg_max(blob, axis):
    return tf.argmax(blob, axis=axis)

register_op('arg_max', _arg_max)

def _neg_relu(tensor):
    return tf.minimum(tf.constant(0, dtype=tf.float32), tensor)

register_op('neg_relu', _neg_relu)
