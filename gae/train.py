from __future__ import division
from __future__ import print_function

import time
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import scipy.sparse as sp

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from gae.optimizer import OptimizerAE, OptimizerVAE
from gae.input_data import load_data
from gae.model import GCNModelAE, GCNModelVAE
from gae.preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges
DEBUG = True
# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 16, 'Number of units in hidden layer 2.')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')

flags.DEFINE_string('model', 'gcn_ae', 'Model string.')
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')
flags.DEFINE_string('tflog', '/home/ioannis/repos/vgae_logs/', 'tflog.')
flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')

model_str = FLAGS.model
dataset_str = FLAGS.dataset
from datetime import datetime
current_time = datetime.now().strftime('%b%d_%H-%M-%S')
tflogs =FLAGS.tflog +current_time+"original_tf"+'/'
if not os.path.exists(tflogs):
    os.makedirs(tflogs)
    
# Load data
adj, features = load_data(dataset_str)
if DEBUG:
    print(f"adj type, {type(adj)}")
    print(f"adj.shape, {adj.shape}")
    print(f"adj[:10,:10], {adj[:10, :10]}")
    print(f"adj {adj}")
    print(f"feature.shape, {features.shape}")
# Store original adjacency matrix (without diagonal entries) for later
adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
if DEBUG:
    print(f"ad_orig type, {type(adj_orig)}")
    print(f"adj_orig.shape, {adj_orig.shape}")

adj_orig.eliminate_zeros()

adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj, random_seed=42)
if DEBUG:
    print(f"adj_train type, {type(adj_train)}")
    print(f"adj_train shape, {type(adj_train.shape)}")
    print(f"train_edges type, {type(train_edges)}")
    print('*'*20)
    print(f"train_edges shape, {train_edges.shape}")
    print(f"val_edges shape, {val_edges.shape}")
    print(f"test_edges shape, {test_edges.shape}")
    print('*'*20)
    print(f"val_edges_false type, {type(val_edges_false)}")
    print(f"test_edges_false type, {type(test_edges_false)}")
    print(f"len val edges false, {len(val_edges_false)}")
    print(f"len test edges false, {len(test_edges_false)}")
    print('*'*20)
    print(f"val_edges[:20], {val_edges[:20]}")
    print(f"test_edges[:20] type, {test_edges[:20]}")
    print(f"val_edges_false[:20], {val_edges_false[:20]}")
    print(f"test_edges_false[:20] type, {test_edges_false[:20]}")
    print('*'*20)
    
    print(f"train_edges[:2], {train_edges[:2]}")
adj = adj_train

if FLAGS.features == 0:
    features = sp.identity(features.shape[0])  # featureless

# Some preprocessing
adj_norm = preprocess_graph(adj)

# Define placeholders
placeholders = {
    'features': tf.sparse_placeholder(tf.float32),
    'adj': tf.sparse_placeholder(tf.float32),
    'adj_orig': tf.sparse_placeholder(tf.float32),
    'dropout': tf.placeholder_with_default(0., shape=())
}

num_nodes = adj.shape[0]

features = sparse_to_tuple(features.tocoo())
num_features = features[2][1]
features_nonzero = features[1].shape[0]

# Create model
model = None
if model_str == 'gcn_ae':
    model = GCNModelAE(placeholders, num_features, features_nonzero)
elif model_str == 'gcn_vae':
    model = GCNModelVAE(placeholders, num_features, num_nodes, features_nonzero)

pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

# Optimizer
with tf.name_scope('optimizer'):
    if model_str == 'gcn_ae':
        opt = OptimizerAE(preds=model.reconstructions,
                          labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                      validate_indices=False), [-1]),
                          pos_weight=pos_weight,
                          norm=norm)
    elif model_str == 'gcn_vae':
        opt = OptimizerVAE(preds=model.reconstructions,
                           labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                       validate_indices=False), [-1]),
                           model=model, num_nodes=num_nodes,
                           pos_weight=pos_weight,
                           norm=norm)

# Initialize session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

cost_val = []
acc_val = []
def vis_a(a, x_label='epoch',y_label='loss', name=None,save_a=True):
    if save_a:
        np.save('./outputs/'+name+'.npy', np.array(a))
    plt.plot(a)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(name)
    plt.savefig(name+'.png')
    plt.close()

def get_roc_score(edges_pos, edges_neg, emb=None):
    if emb is None:
        feed_dict.update({placeholders['dropout']: 0})
        emb = sess.run(model.z_mean, feed_dict=feed_dict)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score

train_loss = []
train_acc = []
val_loss = []

val_roc = []
val_ap = []

adj_label = adj_train + sp.eye(adj_train.shape[0])
adj_label = sparse_to_tuple(adj_label) 
#writer = tf.summary.create_file_writer(tflogs)
#writer = tf.summary.FileWriter(tflogs)
writer = SummaryWriter(log_dir=tflogs)
# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    # Run single weight update
    outs = sess.run([opt.opt_op, opt.cost, opt.accuracy], feed_dict=feed_dict)

    # Compute average loss
    avg_cost = outs[1]
    avg_accuracy = outs[2]
    train_loss.append(avg_cost)
    train_acc.append(avg_accuracy)
    roc_curr, ap_curr = get_roc_score(val_edges, val_edges_false)
    val_roc.append(roc_curr)
    val_ap.append(ap_curr)
    #with train_summary_writer.as_default():
    #tf.summary.scalar('loss', train_loss.result(), step=epoch)
    #train_summary_writer
    writer.add_scalar('Loss/train',avg_cost , epoch)
    writer.add_scalar('ROC/val', roc_curr , epoch)
    writer.add_scalar('AP/val',ap_curr , epoch)
    
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
          "train_acc=", "{:.5f}".format(avg_accuracy), "val_roc=", "{:.5f}".format(val_roc[-1]),
          "val_ap=", "{:.5f}".format(ap_curr),
          "time=", "{:.5f}".format(time.time() - t))

print("Optimization Finished!")
vis_a(train_loss, 'epoch','loss','train_loss_tf_gae_200')
vis_a(train_acc, 'epoch','accuracy','train_accuracy_tf_gae_200')
vis_a(val_roc, 'epoch','roc','val_roc_tf_gae_200')
vis_a(val_ap, 'epoch','ap','val_ap_tf_gae_200')
roc_score, ap_score = get_roc_score(test_edges, test_edges_false)
print('Test ROC score: ' + str(roc_score))
print('Test AP score: ' + str(ap_score))
