import netdef_slim as nd


def softmax2_soft_translator(data):
    return nd.ops.slice(nd.ops.softmax(data), 1)[1]

def softmax2_hard_translator(data):
    return nd.ops.slice(nd.ops.threshold(nd.ops.softmax(data), 0.5), 1)[1]

def iul_b_log_translator(value):
    return nd.ops.exp(value, scale=1)

def iul_b_log_ent_translator(value):
    value2 = nd.ops.mul(value, nd.ops.const_like(value, 2.0))
    value2_eps = nd.ops.add(value2, nd.ops.const_like(value2, 1e-4))
    log = nd.ops.log(value2_eps)
    ent_x, ent_y = nd.ops.slice(nd.ops.add(log, nd.ops.const_like(log, 1.0)), 1)
    ent = nd.ops.add(ent_x, ent_y)
    return ent