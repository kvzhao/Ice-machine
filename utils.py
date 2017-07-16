import tensorflow as tf
import tensorflow.contrib.layers as layers

def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)

def leaky_relu_batch_norm(x, alpha=0.2):
    return leaky_relu(layers.batch_norm(x), alpha)

def relu_batch_norm(x):
    return tf.nn.relu(layers.batch_norm(x))

def pdb(s, d, l):
    return ((s+d) % l + l) % l

def spinization (bx, threshold=.75):
    bx[bx >= +threshold] = 1.0
    bx[bx <= -threshold] = -1.0
    return bx

def cal_energy(bx):
    shape = bx.shape
    batch_size = shape[0]
    w, h, c = shape[1], shape[2], shape[3]
    N = w * h
    total_mean_eng = 0.0
    for b in range(batch_size):
        # ignore channel
        s = bx[b,:,:,0]
        eng = 0.0
        for j in range(w):
            for i in range(h):
                se = 0.0
                ip = pdb(i, +1, h)
                im = pdb(i, -1, h)
                jp = pdb(j, +1, w)
                jm = pdb(j, -1, w)
                if ((i+j) % 2 == 0):
                    se = float(s[ip][j]+s[i][jm] + s[im][j] + s[i][jp]\
                            + s[ip][jp] + s[ip][jp])
                else:
                    se = float(s[ip][j]+s[i][jm] + s[im][j] + s[i][jp]\
                            + s[ip][jm] + s[im][jp])
                se *= s[i,j]
                eng += se
        total_mean_eng += (eng/N/2.0)
    return total_mean_eng / batch_size

def cal_defect_density(bx):
    shape = bx.shape
    batch_size = shape[0]
    w, h, c = shape[1], shape[2], shape[3]
    N = w * h
    total_defect_den = 0.0
    for b in range(batch_size):
        # ignore channel
        s = bx[b,:,:,0]
        dd = 0.0
        for j in range(w):
            for i in range(h):
                se = 0.0
                ip = pdb(i, +1, h)
                im = pdb(i, -1, h)
                jp = pdb(j, +1, w)
                jm = pdb(j, -1, w)
                if ((i+j) % 2 == 0):
                    se = float(s[i][j] + s[i][jp] + s[ip][j] + s[ip][jp])
                dd += abs(se)
        total_defect_den += (dd/N/2.0)
    return total_defect_den / batch_size

def variable_summaries(var):
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)