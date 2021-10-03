import numpy as np
import tensorflow as tf
import input
from tensorflow.contrib.layers import convolution2d as conv
from tensorflow.contrib.layers import convolution2d_transpose as conv_trans
from tensorflow.contrib.layers import fully_connected as full
from tensorflow.contrib.layers import max_pool2d as maxp
# from tensorflow.contrib.layers import avg_pool2d as maxp
from tensorflow.contrib.layers import softmax
import scipy.io as scio
import Sparse
from tensorflow.contrib.layers import avg_pool2d as average
import matplotlib.pyplot as plt

all_data = scio.loadmat('./data/data.mat')['data'][:,:,-1] #[29655,2048,1]
all_label = scio.loadmat('./data/label.mat')['label'] #[n,1]
all_data = np.squeeze(all_data)
data = all_data[:10800,:]
labels = all_label[:10800]
test_data = all_data[10800:13500,:]
test_labels = all_label[10800:13500]
data_sp = data*2-1
test_data_sp = test_data*2-1
pack_sp = input.read_data_sets(data_sp, labels, test_data_sp, test_labels, one_hot=True)
pack = input.read_data_sets(data, labels, test_data, test_labels, one_hot=True)

with tf.variable_scope('model'):
    x = tf.placeholder(tf.float32, [None, 2048], name='x')
    y = tf.placeholder(tf.float32, [None,9])
    x1 = tf.reshape(x, [-1, 2048, 1, 1])

    conv1 = conv(x1, 32, [3, 1], [1, 1])
    sp_in = tf.reshape(conv1,[-1, 32])
    Shape_spa = int(sp_in.shape[1])
    selected_x, sp_loss1, s_weight = Sparse.sparse_selection(sp_in, Shape_spa)
    selected_x = tf.reshape(selected_x,[-1,2048])
    weight = tf.constant(50, dtype=tf.float32, shape=[2048], name='Const')
    selected_x = tf.multiply(selected_x, weight)
    conv_in = tf.reshape(selected_x,[-1, 2048, 1, 1])
    conv1 = conv(conv_in, 32, [64, 1], [2, 1])
    maxp1 = maxp(conv1,[2,1],[2,1])
    conv2 = conv(maxp1, 64, [3, 1],[2, 1])
    maxp2 = maxp(conv2,[2,1],[2,1])
    feature = tf.identity(maxp2,name='feature')
    unpool2 = tf.image.resize_images(maxp2,size=(256,1))
    deconv2 = conv_trans(unpool2,32,[3,1],[2,1])

    conv_res = maxp1
    squeeze_avg = average(conv_res, [conv_res.shape[1], conv_res.shape[2]], stride=1)
    squeeze_max = maxp(conv_res, [conv_res.shape[1], conv_res.shape[2]], stride=1)
    squeeze = squeeze_avg + squeeze_max
    flat = tf.reshape(squeeze, [-1, int(squeeze.shape[3])])
    excitation = full(flat, int(int(squeeze.shape[3]) / 2),activation_fn=tf.nn.relu)
    dim = int(conv_res.shape[3])
    co_in = tf.reshape(excitation,[-1, excitation.shape[1],1,1])
    co = conv(co_in, dim, [3, 1],[1, 1],activation_fn=tf.nn.relu)
    avg = average(co, [co.shape[1], co.shape[2]], stride=1)
    avg = tf.nn.relu(avg)
    excitation2 = tf.reshape(avg, [-1, 1, 1, dim])
    selected = conv_res * excitation2
    res = selected + deconv2
    unpool1 = tf.image.resize_images(res,size=(1024,1))
    deconv1 = conv_trans(unpool1,1,[64,1],[2,1])
    output = tf.reshape(deconv1, [-1, 2048])


with tf.variable_scope('sparse'):
    sp_loss2 = tf.reduce_sum(tf.pow(selected_x - x,2))
    sparse_loss = sp_loss1
    sparse_loss2 = sp_loss2
    loss_0 = sparse_loss * 20 + sparse_loss2 * 0.5
    optimizer0 = tf.train.AdamOptimizer(learning_rate=0.005).minimize(loss_0)
    sparse_lo, sparse_lo2 = [], []

with tf.variable_scope('unsupervised'):
    loss_1 = tf.reduce_mean(tf.pow(x - output, 2))
    optimizer1 = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss_1)
    mse = []

with tf.variable_scope('fine-tuning'):
    full1 = full(tf.reshape(feature, [-1, 128*64]), 200, activation_fn=tf.nn.relu)
    full2 = full(full1, 9, activation_fn=None)
    logits = softmax(full2)
    loss_2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=full2, labels=y))
    optimizer2 = tf.train.AdamOptimizer(0.0005).minimize(loss_2)
    aa = tf.argmax(logits, 1)
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    train_acc, valid_acc, cross_entropy , valid_Lo = [], [], [], []
# #
# '''
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        batch_sp = pack_sp.train.next_batch(50)
        sp_loss, sp_loss2,_ = sess.run([sparse_loss, sparse_loss2, optimizer0], feed_dict={x: batch_sp[0]})
        sparse_lo.append(sp_loss)
        sparse_lo2.append(sp_loss2)
        if (i+1)%50 == 0:
            print('step:%d,Sparse loss=%g,Sparse loss2=%g'%(i,sp_loss, sp_loss2))
    saver = tf.train.Saver()
    saver.save(sess, './result/model_sp')
    print('sparse finished')


    for i in range(1000):
        batch = pack.train.next_batch(50)
        batch_loss1, _ = sess.run([loss_1, optimizer1], feed_dict={x: batch[0]})
        mse.append(batch_loss1)
        if (i+1)%50 == 0:
            print('step:%d, loss:%g'%(i,batch_loss1))
    saver = tf.train.Saver()
    saver.save(sess, './result/model_unsupervised')
    print('unsupervised finished')

    for i in range(2000):
        batch = pack.train.next_batch(50)
        train_accuracy, batch_loss2, _ = sess.run([accuracy, loss_2, optimizer2],
                                                 feed_dict={x: batch[0], y: batch[1]})
        train_acc.append(train_accuracy)
        valid_accuracy, valid_loss= sess.run([accuracy, loss_2], feed_dict={x: pack.validation.images, y: pack.validation.labels})
        valid_acc.append(valid_accuracy)
        cross_entropy.append(batch_loss2)
        valid_Lo.append(valid_loss)
        sparse_lo.append(sp_loss)
        if i%50==0:
            print("For Step%d:\nTraining loss=%g,Training accuracy=%g,Validtion accuracy=%g"
                  % (i, batch_loss2, train_accuracy, valid_accuracy))

    test_acc = sess.run(accuracy, feed_dict={x: pack.test.images, y: pack.test.labels})
    print("test accuracy is %g" % test_acc)

    y_predict_index = sess.run( aa, feed_dict={x: pack.test.images, y: pack.test.labels})

    saver = tf.train.Saver()
    saver.save(sess, './result/SRCAE')
    np.save("./result/train_acc.npy", train_acc)
    np.save("./result/valid_acc.npy", valid_acc)
    np.save("./result/cross_entropy.npy", cross_entropy)
    np.save("./result/valid_loss.npy", valid_Lo)
    np.save("./result/mse.npy", mse)
    np.save("./result/sparse_loss.npy", sparse_lo)




sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess,'./result/model_sp')
plt.rc('font',family='Times New Roman')
num = 21
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
batch = pack_sp.train.next_batch(40)
input = batch[0][num]
input = np.reshape(input,[-1,2048])
label = batch[1][num]
out = sess.run(selected_x, feed_dict={x: input})
plt.subplot(2,1,1)
plt.plot(input[0,:],'b',linewidth=1)
plt.xlim(0, 2048)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.title("Original signal",fontsize=23)
plt.subplot(2,1,2)
plt.plot(out[0,:],'r',linewidth=1)
plt.xlim(0,2048)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title("Sparse representation",fontsize=23)
plt.tight_layout()
plt.subplots_adjust(wspace=0.3, hspace=0.4)
plt.savefig('./result/SR.png', format='png')
plt.show()


sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess,'./result/model_unsupervised')
plt.rc('font',family='Times New Roman')
num = 21
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
batch = pack.train.next_batch(40)
input = batch[0][num]
input = np.reshape(input,[-1,2048])
out = sess.run(output, feed_dict={x: input})
plt.subplot(2,1,1)
plt.plot(input[0, :],'b',linewidth=1)
plt.xlim(0, 2048)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.title("Original signal",fontsize=23)
plt.subplot(2,1,2)
plt.plot(out[0, :],'r',linewidth=1)
plt.xlim(0,2048)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title("Reconstructed signal",fontsize=23)
plt.tight_layout()
plt.subplots_adjust(wspace=0.3, hspace=0.4)
plt.savefig('./result/Reconstructed signal.png', format='png')
plt.show()






