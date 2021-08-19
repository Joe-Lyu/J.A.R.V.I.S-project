import random
import os
import tensorflow.compat.v1 as tf
import tempfile

import twremat

def splice_op(op, input_map, control_inputs=None):
    g = op.graph
    node_def = tf.NodeDef()
    node_def.CopyFrom(op.node_def)
    node_def.name = g.unique_name(op.name + '_copy')
    inputs = [input_map.get(x, x) for x in op.inputs]
    new_control_inputs = [input_map.get(x, x) for x in op.control_inputs]
    if control_inputs:
        new_control_inputs.extend([x for x in control_inputs if x is not None])
    # new_control_inputs = control_inputs
    output_types = [o.dtype for o in op.outputs]
    op_def = op.op_def
    return tf.Operation(node_def, g, inputs=inputs, output_types=output_types, op_def=op_def, control_inputs=new_control_inputs)

def splice_tensor(ten, new_op):
    i = ten.op.outputs.index(ten)
    return new_op.outputs[i]

def splice(obj, input_map, control_inputs=None):
    if type(obj) is tf.Operation:
        return splice_op(obj, input_map, control_inputs=control_inputs)
    elif type(obj) is tf.Tensor:
        return splice_tensor(obj, input_map.get(obj.op, obj.op))
    elif type(obj) is tf.IndexedSlices:
        return tf.IndexedSlices(values=input_map.get(obj.values, obj.values),
                                indices=input_map.get(obj.indices, obj.indices),
                                dense_shape=input_map.get(obj.dense_shape, obj.dense_shape))
    else:
        raise AssertionError(f'Could not get deps from{repr(type(obj))} {repr(obj)}')

def product(xs):
    r = 1
    for x in xs:
        r *= x
    return r

def shape_size(shape):
    if shape.rank is None:
        return 16
    shape = shape.as_list()
    for i in range(len(shape)):
        if shape[i] is None and i == 0:
            shape[i] = 1
        elif shape[i] is None:
            shape[i] = 1024
    return product(shape)

def graph_from_dfs(deps, starts):
    visited = set()
    frontier = starts
    while frontier:
        x = frontier.pop()
        if x in visited:
            continue
        visited.add(x)
        frontier.extend(list(deps(x)))
    return {x : list(deps(x)) for x in visited}

def get_deps(obj):
    if type(obj) is tf.Operation:
        return list(obj.inputs) + list(obj.control_inputs)
    elif type(obj) is tf.Tensor:
        return [obj.op]
    elif type(obj) is tf.IndexedSlices:
        return [obj.indices, obj.values, obj.dense_shape]
    else:
        raise AssertionError(f'Could not get deps from{repr(type(obj))} {repr(obj)}')


def tensor_graph(compute):
    return graph_from_dfs(get_deps, list(compute))

def blacklist(obj):
    if type(obj) is tf.Operation:
        if 'Assign' in obj.type or 'Variable' in obj.type or 'Placeholder' in obj.type:
            # TODO: Should we do special accounting for
            # ReadVariableOp? Currently we forbid cloning altogether,
            # but it's actually ok to clone this op as long as it
            # doesn't float across an effectful op (Assign). Also
            # currently we don't account for the memory used by
            # ReadVariableOp (is it copy-on-write?).
            # https://www.tensorflow.org/api_docs/python/tf/raw_ops/ReadVariableOp?hl=uk
            return True
    elif type(obj) is tf.Tensor:
        return blacklist(obj.op)
    return False

def estimate_cpu(op):
    return sum(4 * shape_size(t.shape) for t in op.inputs if type(t) is tf.Tensor) + sum(4 * shape_size(t.shape) for t in op.outputs)

def estimate_mem(op):
    return sum(4 * shape_size(t.shape) for t in op.outputs)

def info(op):
    if blacklist(op):
        return {'type': 'effectful'}
    elif type(op) is tf.Operation:
        if 'Reshape' in op.type:
            return {'type': 'pointer'}
        return {'type': 'normal',
                'cpu': estimate_cpu(op),
                'mem': estimate_mem(op)}
    elif type(op) is tf.Tensor:
        return {'type': 'pointer'}
    elif type(op) is tf.IndexedSlices:
        return {'type': 'pointer'}
    else:
        raise AssertionError(repr((type(op), op)))


# Helper functions to flatten and unflatten nested structures of
# tensors and ops so that tf_remat can be applied to structures
# without fiddly marshalling.
def get_ops(compute):
    output = []
    stack = [compute]
    while stack:
        top = stack.pop()
        if type(top) is dict:
            for v in top.values():
                stack.append(v)
        elif type(top) in (list, tuple):
            stack.extend(top)
        elif type(top) in (tf.Operation, tf.Tensor, tf.IndexedSlices):
            output.append(top)
    return output

def replace_ops(top, live):
    if type(top) in (tf.Operation, tf.Tensor, tf.IndexedSlices):
        return live[top]
    elif type(top) is dict:
        return {k : replace_ops(v, live) for (k,v) in top.items()}
    elif type(top) is list:
        return [replace_ops(v, live) for v in top]
    elif type(top) is tuple:
        return tuple(replace_ops(v, live) for v in top)
    else:
        return top


def tf_remat(compute, memlimit):
    compute_ops = get_ops(compute)
    tf_deps = tensor_graph(compute_ops)

    # Relabel with integers
    from_op = {op : i for (i, op) in enumerate(tf_deps.keys())}
    from_node = {i : op for (op, i) in from_op.items()}
    nodes = set(from_node.keys())
    node_deps = {n : [from_op[d] for d in tf_deps[from_node[n]]] for n in nodes}

    node_info = {}
    for n in nodes:
        node_info[n] = info(from_node[n])
        node_info[n]['deps'] = [from_op[d] for d in tf_deps[from_node[n]]]

    steps = twremat.runtwremat(node_info, memlimit, {from_op[c] for c in compute_ops})

    print('Constructing tensorflow graph...')
    live = {}
    last_op = None
    for (action, n) in steps:
        base = from_node[n]
        if action == 'compute':
            input_map = {d : live[d] for d in tf_deps[base] if live[d] != d}
            if blacklist(base) and not input_map:
                live[base] = base
            else:
                live[base] = splice(base, input_map, control_inputs=[last_op])
            if type(base) is tf.Operation:
                last_op = live[base]
        elif action == 'free':
            del live[base]

    return replace_ops(compute, live)
