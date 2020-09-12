from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat')
Genotype_normal = namedtuple('Genotype_normal', 'normal normal_concat')

PRIMITIVES = [
    'none',
    'skip_connect',
    'conv_1x1',
    'Conv1',
    # 'Conv2',
    # 'Conv3',
    # 'Conv4',
    # 'Conv5',
]

# PRIMITIVES = [
#     'none',
#     'skip_connect',
#     'conv_1x1',
#     'edge_conv',
#     'mr_conv',
#     'gat',
#     'semi_gcn',
#     'gin',
#     'sage',
#     'res_sage',
# ]

# ****************************  SGAS CRITERION 1  ****************************** #
Cri1_ModelNet_1 = Genotype(normal=[('conv_1x1', 0), ('mr_conv', 1), ('edge_conv', 0), ('mr_conv', 1), ('conv_1x1', 0), ('gat', 1)], normal_concat=range(1, 5))
Cri1_ModelNet_2 = Genotype(normal=[('mr_conv', 0), ('mr_conv', 1), ('mr_conv', 0), ('edge_conv', 1), ('mr_conv', 0), ('mr_conv', 3)], normal_concat=range(1, 5))
Cri1_ModelNet_3 = Genotype(normal=[('mr_conv', 0), ('gat', 1), ('mr_conv', 1), ('edge_conv', 2), ('conv_1x1', 0), ('conv_1x1', 3)], normal_concat=range(1, 5))
Cri1_ModelNet_4 = Genotype(normal=[('gat', 0), ('res_sage', 1), ('conv_1x1', 0), ('conv_1x1', 1), ('conv_1x1', 2), ('mr_conv', 3)], normal_concat=range(1, 5))
Cri1_ModelNet_5 = Genotype(normal=[('mr_conv', 0), ('mr_conv', 1), ('skip_connect', 0), ('mr_conv', 1), ('mr_conv', 2), ('mr_conv', 3)], normal_concat=range(1, 5))
Cri1_ModelNet_6 = Genotype(normal=[('edge_conv', 0), ('skip_connect', 1), ('conv_1x1', 0), ('conv_1x1', 2), ('skip_connect', 1), ('gat', 2)], normal_concat=range(1, 5))
Cri1_ModelNet_7 = Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('mr_conv', 0), ('mr_conv', 1), ('mr_conv', 1), ('edge_conv', 2)], normal_concat=range(1, 5))
Cri1_ModelNet_8 = Genotype(normal=[('mr_conv', 0), ('mr_conv', 1), ('mr_conv', 1), ('conv_1x1', 2), ('skip_connect', 0), ('gin', 1)], normal_concat=range(1, 5))
Cri1_ModelNet_9 = Genotype(normal=[('edge_conv', 0), ('skip_connect', 1), ('mr_conv', 0), ('mr_conv', 2), ('gin', 1), ('conv_1x1', 3)], normal_concat=range(1, 5))
Cri1_ModelNet_10 = Genotype(normal=[('mr_conv', 0), ('gat', 1), ('mr_conv', 0), ('res_sage', 2), ('mr_conv', 2), ('mr_conv', 3)], normal_concat=range(1, 5))
Cri1_ModelNet_Best = Cri1_ModelNet_9

# ****************************  SGAS CRITERION 2  ****************************** #
Cri2_ModelNet_1 = Genotype(normal=[('skip_connect', 0), ('edge_conv', 1), ('mr_conv', 1), ('gat', 2), ('edge_conv', 1), ('edge_conv', 3)], normal_concat=range(1, 5))
Cri2_ModelNet_2 = Genotype(normal=[('edge_conv', 0), ('edge_conv', 1), ('skip_connect', 1), ('gin', 2), ('mr_conv', 1), ('edge_conv', 2)], normal_concat=range(1, 5))
Cri2_ModelNet_3 = Genotype(normal=[('edge_conv', 0), ('mr_conv', 1), ('edge_conv', 1), ('mr_conv', 2), ('conv_1x1', 0), ('mr_conv', 1)], normal_concat=range(1, 5))
Cri2_ModelNet_4 = Genotype(normal=[('edge_conv', 0), ('mr_conv', 1), ('skip_connect', 1), ('gat', 2), ('conv_1x1', 0), ('gat', 3)], normal_concat=range(1, 5))
Cri2_ModelNet_5 = Genotype(normal=[('edge_conv', 0), ('mr_conv', 1), ('mr_conv', 1), ('gat', 2), ('edge_conv', 0), ('edge_conv', 3)], normal_concat=range(1, 5))
Cri2_ModelNet_6 = Genotype(normal=[('edge_conv', 0), ('mr_conv', 1), ('mr_conv', 1), ('mr_conv', 2), ('gin', 1), ('mr_conv', 3)], normal_concat=range(1, 5))
Cri2_ModelNet_7 = Genotype(normal=[('mr_conv', 0), ('skip_connect', 1), ('mr_conv', 0), ('conv_1x1', 1), ('edge_conv', 0), ('edge_conv', 1)], normal_concat=range(1, 5))
Cri2_ModelNet_8 = Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('edge_conv', 0), ('mr_conv', 2), ('mr_conv', 0), ('mr_conv', 2)], normal_concat=range(1, 5))
Cri2_ModelNet_9 = Genotype(normal=[('mr_conv', 0), ('mr_conv', 1), ('mr_conv', 1), ('gin', 2), ('conv_1x1', 0), ('conv_1x1', 1)], normal_concat=range(1, 5))
Cri2_ModelNet_10 = Genotype(normal=[('edge_conv', 0), ('mr_conv', 1), ('edge_conv', 1), ('edge_conv', 2), ('mr_conv', 0), ('edge_conv', 1)], normal_concat=range(1, 5))
Cri2_ModelNet_Best = Cri2_ModelNet_4

# ****************************  Atom Search   ****************************** #
Cri2_ModelNet_Atom = Genotype(normal=[('skip_connect', 0), ('conv_1x1', 1), ('conv_1x1', 0), ('Conv1', 1), ('skip_connect', 0), ('Conv1', 1)], normal_concat=range(1, 5))
Cri2_ModelNet_Atom_Best = Cri2_ModelNet_Atom

# ****************************  Random Search   ****************************** #
random_ModelNet_1 = Genotype(normal=[('gin', 0), ('mr_conv', 1), ('res_sage', 0), ('res_sage', 2), ('edge_conv', 3), ('gin', 0)], normal_concat=range(1, 5))
random_ModelNet_2 = Genotype(normal=[('sage', 0), ('mr_conv', 1), ('mr_conv', 1), ('gat', 0), ('gin', 0), ('gat', 1)], normal_concat=range(1, 5))
random_ModelNet_3 = Genotype(normal=[('sage', 1), ('conv_1x1', 0), ('mr_conv', 0), ('gat', 2), ('gin', 0), ('res_sage', 3)], normal_concat=range(1, 5))
random_ModelNet_4 = Genotype(normal=[('gin', 0), ('gat', 1), ('conv_1x1', 0), ('semi_gcn', 2), ('sage', 0), ('mr_conv', 1)], normal_concat=range(1, 5))
random_ModelNet_5 = Genotype(normal=[('conv_1x1', 1), ('res_sage', 0), ('gin', 0), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 1)], normal_concat=range(1, 5))
random_ModelNet_6 = Genotype(normal=[('gin', 1), ('mr_conv', 0), ('semi_gcn', 2), ('gat', 0), ('skip_connect', 2), ('res_sage', 0)], normal_concat=range(1, 5))
random_ModelNet_7 = Genotype(normal=[('sage', 0), ('gat', 1), ('sage', 2), ('semi_gcn', 0), ('gat', 3), ('gin', 1)], normal_concat=range(1, 5))
random_ModelNet_8 = Genotype(normal=[('sage', 0), ('skip_connect', 1), ('mr_conv', 1), ('edge_conv', 2), ('gat', 2), ('skip_connect', 3)], normal_concat=range(1, 5))
random_ModelNet_9 = Genotype(normal=[('edge_conv', 1), ('edge_conv', 0), ('semi_gcn', 1), ('conv_1x1', 0), ('gat', 0), ('mr_conv', 2)], normal_concat=range(1, 5))
random_ModelNet_10 = Genotype(normal=[('conv_1x1', 1), ('edge_conv', 0), ('gin', 2), ('gat', 0), ('gat', 3), ('conv_1x1', 0)], normal_concat=range(1, 5))
Random_ModelNet_Best = random_ModelNet_9

att = [[0.7993646, 0.7475484, -0.5459754, -0.54298896, -0.61250013], [-0.8690626, -0.11340748, 0.43529555, 1.2458676, -0.7564482], [0.44827592, 0.4484686, 0.44857126, 0.44897148, -0.6752271]]
