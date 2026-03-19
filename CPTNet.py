## General Imports
import numpy as np
#import cv2
from numpy import asarray
import  scipy.ndimage as sn
import math
import scipy.io

## Tensorflow imports
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Layer,Dense,LayerNormalization,Input,Flatten,Softmax
from tensorflow.keras.layers import Concatenate,Multiply,Conv3D,LeakyReLU
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

##Others
from sklearn.decomposition import PCA
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau, Callback
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score

def PCA_fit_transform(XData,ncmp):
    pca_obj=PCA(n_components=ncmp)
    xdata_shape=XData.shape
    XData=XData.reshape(-1,xdata_shape[-1])
    pca_obj.fit(XData)
    XData=pca_obj.transform(XData)
    XData=XData.reshape(xdata_shape[0],xdata_shape[1],xdata_shape[2],XData.shape[-1])
    return XData,pca_obj
## Shufling
def unison_shuffled_copies(a, b, c):
    assert len(a) == len(b) == len(c)
    p = np.random.permutation(len(a))
    return a[p], b[p], c[p]
##
def horizontal_flipping_v2(XTrain):
    XHTrain=np.zeros((XTrain.shape))
    for i in range(len(XTrain)):
        for pc in range(XTrain.shape[-2]):
            for bn in range(XTrain.shape[-1]):
                XHTrain[i,:,:,pc,bn]=np.fliplr(np.copy(XTrain[i,:,:,pc,bn]))
    return XHTrain
def vertical_flipping_v2(XTrain):
    XHTrain=np.zeros((XTrain.shape))
    for i in range(len(XTrain)):
        for pc in range(XTrain.shape[-2]):
            for bn in range(XTrain.shape[-1]):
                XHTrain[i,:,:,pc,bn]=np.flipud(np.copy(XTrain[i,:,:,pc,bn]))
    return XHTrain


##
def cal_SAM_mas_Xformer(XTrain):
    max_angle=0
    ip_shape=XTrain.shape
    patch_width=ip_shape[2]
    mx_dis=(patch_width-1)/2
    cp_pos=patch_width//2
    SAM_masks=np.zeros((ip_shape[0],patch_width,patch_width))
    for smpl in range(ip_shape[0]):
        for rw in range(patch_width):
            for cl in range(patch_width):
                if(rw==cp_pos and cl==cp_pos):
                    SAM_masks[smpl,rw,cl]=1
                    continue
                chb_dist=max(abs(mx_dis-rw),abs(mx_dis-cl))
                wt_dist=(mx_dis+1-chb_dist)/(mx_dis+1)
                mask_wgt=wt_dist
                SAM_masks[smpl,rw,cl]=mask_wgt if(mask_wgt>0) else 0
    return SAM_masks


def dataset_preparation(patch_size):
    extra_width=patch_size//2
    temp_pad=np.pad(temp,((extra_width,extra_width),(extra_width,extra_width),(0,0),(0,0)),mode='edge')
    gt_temp_pad=np.pad(gt_temp,((extra_width,extra_width),(extra_width,extra_width)),mode='edge')
    # temp_og_pad=np.pad(temp_og,((extra_width,extra_width),(extra_width,extra_width),(0,0)),mode='edge')
    print(temp_pad.shape,temp.shape,gt_temp_pad.shape)
    ## Patches Extracting
    i_h=temp_pad.shape[0]
    i_w=temp_pad.shape[1]
    X_Images=list()
    Y_Labels=list()
    # X_og=list()
    for hi in range(extra_width,i_h-extra_width):
        for wi in range(extra_width,i_w-extra_width):
            if(gt_temp_pad[hi][wi]==0):
                continue
            h_start=hi-extra_width
            w_start=wi-extra_width
            h_end=hi+extra_width
            w_end=wi+extra_width
            mini_patch=temp_pad[h_start:h_end+1,w_start:w_end+1,:,:]
            mini_class=gt_temp_pad[hi][wi]
            # mini_og=temp_og_pad[h_start:h_end+1,w_start:w_end+1,:]
            X_Images.append(mini_patch)
            Y_Labels.append(mini_class)
            # X_og.append(mini_og)
    X_Images=np.asarray(X_Images)
    Y_Labels=np.asarray(Y_Labels)
    # X_og=np.asarray(X_og)

    X_Images,Y_Labels,Y_dummy=unison_shuffled_copies(X_Images,Y_Labels,Y_Labels)
    ## Removing Background samples
    ##
    # Trn_IP=[0,10,71,41,11,24,36,10,23,10,48,122,29,10,63,19,10]
    Trn_IP_counter=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    X_Train=list()
    Y_Train=list()
    X_Test=list()
    Y_Test=list()
    # X_og_Tr=list()
    # X_og_Ts=list()
    ##
    for i in range(len(X_Images)):
        cls_i=Y_Labels[i]
        if(Trn_IP_counter[cls_i]<Trn_IP[cls_i]):
            X_Train.append(X_Images[i])
            Y_Train.append(Y_Labels[i])
            # X_og_Tr.append(X_og[i])
            Trn_IP_counter[cls_i]=Trn_IP_counter[cls_i]+1
        else:
            X_Test.append(X_Images[i])
            Y_Test.append(Y_Labels[i])
            # X_og_Ts.append(X_og[i])
    ##Free Up Ram
    del X_Images
    del Y_Labels
    # del X_og
    del gt_temp_pad
    del temp_pad
    ##
    X_Train=np.asarray(X_Train)
    Y_Train=np.asarray(Y_Train)
    X_Test=np.asarray(X_Test)
    Y_Test=np.asarray(Y_Test)
    # X_og_Tr=np.asarray(X_og_Tr)
    # X_og_Ts=np.asarray(X_og_Ts)
    print(X_Train.shape,Y_Train.shape,Y_Test.shape,X_Test.shape)
    # print(X_og_Tr.shape,X_og_Ts.shape)
    ## Validation dataset preparation
    # Vld_IP=[0,10,71,41,11,24,36,10,23,5,48,122,29,10,63,19,10]
    Vld_IP_counter=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    X_Vld=list()
    Y_Vld=list()
    # X_og_Vld=list()
    New_X_Test=list()
    New_Y_Test=list()
    # New_X_og=list()
    for i in range(len(X_Test)):
        cls_i=Y_Test[i]
        if(Vld_IP_counter[cls_i]<Vld_IP[cls_i]):
            X_Vld.append(X_Test[i])
            Y_Vld.append(Y_Test[i])
            # X_og_Vld.append(X_og_Ts[i])
            Vld_IP_counter[cls_i]=Vld_IP_counter[cls_i]+1
        else:
            New_X_Test.append(X_Test[i])
            New_Y_Test.append(Y_Test[i])
            # New_X_og.append(X_og_Ts[i])
    del X_Test
    del Y_Test
    # del X_og_Ts
    X_Test=np.asarray(New_X_Test)
    Y_Test=np.asarray(New_Y_Test)
    X_Vld=np.asarray(X_Vld)
    Y_Vld=np.asarray(Y_Vld)
    # X_og_Ts=np.asarray(New_X_og)
    # X_og_Vld=np.asarray(X_og_Vld)
    list_class_length=list()
    for i in range(17):
        x=np.sum(Y_Train==i)
        y=np.sum(Y_Test==i)
        z=np.sum(Y_Vld==i)
        list_class_length.append((x+y+z))
        print(x,z,y)
    print(list_class_length)
    X_og_Tr=cal_SAM_mas_Xformer(X_Train)
    X_og_Ts=cal_SAM_mas_Xformer(X_Test)
    X_og_Vld=cal_SAM_mas_Xformer(X_Vld)

    #Agumentation
    X_Train,Y_Train,X_og_Tr=data_agumentation(X_Train,Y_Train,X_og_Tr)
    permutation = np.random.permutation(len(X_Train))
    X_Train=X_Train[permutation]
    Y_Train=Y_Train[permutation]
    X_og_Tr=X_og_Tr[permutation]
    #
    print(X_Train.shape,Y_Train.shape)
    print(X_Vld.shape,Y_Vld.shape)
    print(X_Test.shape,Y_Test.shape)
    print(X_og_Tr.shape,X_og_Vld.shape,X_og_Ts.shape)
    #hot encoding
    ## Training Datsets
    Y_Tr_hot=Y_Train-1
    Y_Tr_hot=to_categorical(Y_Tr_hot)
    ## Validation Dataset
    Y_Vld_hot=Y_Vld-1
    Y_Vld_hot=to_categorical(Y_Vld_hot)
    ## Testing Data
    #labels
    Y_Ts_hot=Y_Test-1
    Y_Ts_hot=to_categorical(Y_Ts_hot)
    return X_Train,Y_Train,Y_Tr_hot,X_Test,Y_Test,Y_Ts_hot,X_Vld,Y_Vld,Y_Vld_hot,X_og_Tr,X_og_Vld,X_og_Ts

def data_agumentation(X_Train,Y_Train,X_og_Tr):
    X_Train_Copy2=np.copy(X_Train)
    Y_Train_Copy2=np.copy(Y_Train)
    #
    X_hf_Train=horizontal_flipping_v2(X_Train_Copy2)
    print(X_hf_Train.shape)
    X_Train=np.concatenate((X_Train,X_hf_Train),axis=0)
    del X_hf_Train
    #
    X_vf_Train=vertical_flipping_v2(X_Train_Copy2)
    print(X_vf_Train.shape)
    X_Train=np.concatenate((X_Train,X_vf_Train),axis=0)
    del X_vf_Train
    #
    X_r90_Train=np.rot90(np.copy(X_Train_Copy2),k=1,axes=(1,2))
    print(X_r90_Train.shape)
    X_Train=np.concatenate((X_Train,X_r90_Train),axis=0)
    del X_r90_Train
    #
    X_r180_Train=np.rot90(np.copy(X_Train_Copy2),k=2,axes=(1,2))
    print(X_r180_Train.shape)
    X_Train=np.concatenate((X_Train,X_r180_Train),axis=0)
    del X_r180_Train
    #
    X_r270_Train=np.rot90(np.copy(X_Train_Copy2),k=3,axes=(1,2))
    print(X_r270_Train.shape)
    X_Train=np.concatenate((X_Train,X_r270_Train),axis=0)
    del X_r270_Train
    #
    del X_Train_Copy2

    Y_Train=np.concatenate((Y_Train,Y_Train_Copy2),axis=0)
    Y_Train=np.concatenate((Y_Train,Y_Train_Copy2),axis=0)
    Y_Train=np.concatenate((Y_Train,Y_Train_Copy2),axis=0)
    Y_Train=np.concatenate((Y_Train,Y_Train_Copy2),axis=0)
    Y_Train=np.concatenate((Y_Train,Y_Train_Copy2),axis=0)
    del Y_Train_Copy2

    ##Og agumentations
    X_og_Tr_copy_2=np.copy(X_og_Tr)
    X_og_Tr_hf=np.array([np.fliplr(img) for img in X_og_Tr_copy_2])
    X_og_Tr=np.concatenate((X_og_Tr,X_og_Tr_hf),axis=0)
    del X_og_Tr_hf
    #
    X_og_Tr_vf=np.array([np.flipud(img) for img in X_og_Tr_copy_2])
    X_og_Tr=np.concatenate((X_og_Tr,X_og_Tr_vf),axis=0)
    del X_og_Tr_vf
    #
    X_og_Tr_r90=np.rot90(np.copy(X_og_Tr_copy_2),k=1,axes=(1,2))
    X_og_Tr=np.concatenate((X_og_Tr,X_og_Tr_r90),axis=0)
    del X_og_Tr_r90
    #
    X_og_Tr_r180=np.rot90(np.copy(X_og_Tr_copy_2),k=2,axes=(1,2))
    X_og_Tr=np.concatenate((X_og_Tr,X_og_Tr_r180),axis=0)
    del X_og_Tr_r180
    #
    X_og_Tr_r270=np.rot90(np.copy(X_og_Tr_copy_2),k=3,axes=(1,2))
    X_og_Tr=np.concatenate((X_og_Tr,X_og_Tr_r270),axis=0)
    del X_og_Tr_r270
    #
    del X_og_Tr_copy_2
    #return
    return X_Train,Y_Train,X_og_Tr



def hot_to_labels(Y_hot):
    Y_labels=np.zeros((len(Y_hot),1))
    for smpl in range(len(Y_hot)):
        pred_label=np.argmax(Y_hot[smpl])+1
        Y_labels[smpl]=pred_label
    return Y_labels

def cal_class_accuracies(Y_True,Y_Pred,num_classes):
    cc_true=np.zeros((num_classes,1))
    cc_correct=np.zeros((num_classes,1))
    count=0;
    for smpl in range(len(Y_True)):
        lbl=Y_True[smpl]
        if(Y_Pred[smpl]==lbl):
            cc_correct[lbl-1] +=1
            count +=1
        cc_true[lbl-1] +=1
    cls_accuracy=cc_correct/cc_true
    print('Class accuracies')
    print(cls_accuracy)
    print('Avergae accuracy')
    print(np.mean(cls_accuracy))
    print('Overall accuracy')
    print(count/len(Y_True))
    return cls_accuracy,np.mean(cls_accuracy),(count/len(Y_True))

class LearnableMultiplication(Layer):
    def __init__(self, input_dim, output_dim):
        super(LearnableMultiplication, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_regularizer=rl_l1_l2
    def build(self, input_shape):
        # Define a learnable weight matrix of shape (input_dim, output_dim)
        self.B = self.add_weight(
            shape=(self.input_dim, self.output_dim),
            initializer='random_normal',
            trainable=True,
            name='learnable_matrix',
            regularizer=self.kernel_regularizer
        )

    def call(self, inputs):
        # Perform tensordot on the last axis of inputs and first axis of B
        return tf.tensordot(inputs, self.B, axes=[-1, 0])  # result: (..., output_dim)
class MultiHeadAttention(Layer):
    def __init__(self,input_dim,num_heads=5):
        super(MultiHeadAttention, self).__init__()
        # assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        # self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.input_dim=input_dim
        self.head_dim = input_dim // num_heads
        self.key_dim=input_dim // num_heads
        self.key_embed_dim=self.key_dim * num_heads
        self.Softmax=Softmax(axis=-1)
        self.leaklyReLU=LeakyReLU(alpha=0.1)

        # Projections
        self.q_proj = LearnableMultiplication(input_dim=self.input_dim, output_dim=self.key_embed_dim)
        self.k_proj = LearnableMultiplication(input_dim=self.input_dim, output_dim=self.key_embed_dim)
        self.v_proj = LearnableMultiplication(input_dim=self.input_dim, output_dim=self.key_embed_dim)

        # Final output projection
        self.output_proj = LearnableMultiplication(input_dim=self.key_embed_dim, output_dim=self.input_dim)

    def split_heads(self, x):
        # x: (B, H, W, P, key_embed_dim)
        B, H, W, P, D = tf.unstack(tf.shape(x))
        #print(D//self.num_heads)
        x = tf.reshape(x, (B, H, W, P, self.num_heads, D//self.num_heads))
        return tf.transpose(x, perm=[0, 1, 2, 4, 3, 5])  # (B, H, W, heads, P, head_dim)

    def combine_heads(self, x):
        # x: (B, H, W, heads, P, head_dim)
        x = tf.transpose(x, perm=[0, 1, 2, 4, 3, 5])  # (B, H, W, P, heads, head_dim)
        B, H, W, P,c_heads,c_head_dim = tf.unstack(tf.shape(x))
        return tf.reshape(x, (B, H, W, P,c_heads*c_head_dim ))

    def call(self, inputs):
        # inputs: (B, H, W, P, 5)
        Q = self.q_proj(inputs)
        K = self.k_proj(inputs)
        V = self.v_proj(inputs)

        Q = self.split_heads(Q)  # (B, H, W, heads, P, head_dim)
        K = self.split_heads(K)
        V = self.split_heads(V)

        dk = tf.cast(self.key_dim, tf.float32)

        # Attention: (B, H, W, heads, P, P)
        scores = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(dk)
        B_s,H_s,W_s,Hd_s,P1_s,P2_s=tf.unstack(tf.shape(scores))
        scores= tf.reshape(scores,(B_s,H_s*W_s,Hd_s,P1_s,P2_s))
        attn_weights = self.Softmax(scores)
        attn_weights=tf.reshape(attn_weights,(B_s,H_s,W_s,Hd_s,P1_s,P2_s))

        # Apply attention: (B, H, W, heads, P, head_dim)
        attended = tf.matmul(attn_weights, V)

        # Combine heads and project
        concat = self.combine_heads(attended)  # (B, H, W, P, embed_dim)
        output = self.output_proj(concat)      # (B, H, W, P, 5)
        return output

class TransformerFFN(Layer):
    def __init__(self, hidden_dim, output_dim):
        super(TransformerFFN, self).__init__()
        self.dense1 = Dense(hidden_dim, activation='relu',kernel_regularizer=rl_l1_l2,bias_regularizer=rl_l1_l2)
        self.dense2 = Dense(output_dim,kernel_regularizer=rl_l1_l2,bias_regularizer=rl_l1_l2)

    def call(self, inputs):
        # inputs: (H, W, P, V)
        x = self.dense1(inputs)  # (H, W, P, hidden_dim)
        x = self.dense2(x)       # (H, W, P, output_dim)
        return x

class TransSingleUnit(Layer):
    def __init__(self, input_dim,num_heads=5,ffn_hidden_dim=100):
        super(TransSingleUnit, self).__init__()
        self.mha= MultiHeadAttention(input_dim=input_dim,num_heads=num_heads)
        self.ffn=TransformerFFN(hidden_dim=ffn_hidden_dim,output_dim=input_dim)
        self.lnm1=LayerNormalization(epsilon=1e-6)
        self.lnm2=LayerNormalization(epsilon=1e-6)
    def call(self,X):
        mha_output=self.mha(X)
        mha_add_norm=self.lnm1(X+mha_output)
        ffn_output=self.ffn(mha_add_norm)
        ffn_add_norm=self.lnm2(mha_add_norm+ffn_output)
        return ffn_add_norm
class GAPLayer(Layer):
    def __init__(self):
        super(GAPLayer,self).__init__()
    def call(self,X):
        X=tf.reduce_mean(X,axis=[1,2])
        return X

class ReduceSpatial_v2(Layer):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
    def call(self, inputs1,inputs2):
        tns_mul=inputs1*inputs2
        tf_reduce=tf.reduce_mean(tns_mul, axis=[1,2], keepdims=True)
        return tf_reduce
    def get_config(self):
        config = super().get_config()
        return config


class ReduceSpatial_no_SFB(Layer):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
    def call(self, inputs1,inputs2):
        # tns_mul=inputs1*inputs2
        tf_reduce=tf.reduce_mean(inputs1, axis=[1,2], keepdims=True)
        return tf_reduce
    def get_config(self):
        config = super().get_config()
        return config


class PrunableDense(tf.keras.layers.Layer):
    def __init__(self, units, activation=None, target_sparsity=0.2, l1_reg=0.0, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.target_sparsity = target_sparsity  # store sparsity per layer
        self.l1_reg = l1_reg  # strength of L1 penalty

    def build(self, input_shape):
        last_dim = int(input_shape[-1])

        self.w = self.add_weight(
            shape=(last_dim, self.units),
            initializer='glorot_uniform',
            trainable=True,
            name='kernel',
            regularizer=regularizers.L1(self.l1_reg) if self.l1_reg > 0 else None
        )

        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
            name='bias',
            regularizer=regularizers.L1(self.l1_reg) if self.l1_reg > 0 else None
        )

        self.mask = tf.Variable(tf.ones_like(self.w), trainable=False, dtype=tf.float32)

    def call(self, inputs):
        pruned_w = self.w * self.mask
        x = tf.tensordot(inputs, pruned_w, axes=[[-1], [0]]) + self.b
        if self.activation is not None:
            x = self.activation(x)
        return x

    def prune_by_magnitude(self):
        """Use the layer's own target_sparsity"""
        k = int(tf.size(self.w).numpy() * self.target_sparsity)
        if k == 0:
            return
        w_abs = tf.abs(self.w)
        threshold = tf.sort(tf.reshape(w_abs, [-1]))[k]
        new_mask = tf.cast(w_abs > threshold, tf.float32)
        self.mask.assign(new_mask)



#from tensorflow.keras.callbacks import Callback


# Callback just triggers pruning (no sparsity needed here)
class PruningCallback(tf.keras.callbacks.Callback):
    def __init__(self, layers, start_epoch=5, end_epoch=20):
        super().__init__()
        self.layers = layers
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch

    def on_epoch_end(self, epoch, logs=None):
        if self.start_epoch <= epoch <= self.end_epoch:
            for layer in self.layers:
                layer.prune_by_magnitude()
            print(f"[Epoch {epoch+1}] Layers pruned using their own sparsity")




##Model with multiple TU, with only l2 regularization, no using of multi level features
def modelXformer(input_shape,num_heads,trans_levels,num_classes):
    ##
    input_dim=input_shape[-1]
    # embed_dim=input_dim
    ffn_hidden_dim=num_heads*input_dim
    ##
    inputs=Input(shape=input_shape)
    inputs2=Input(shape=input_shape)
    ## Dense and Pruning
    X_inputs = PrunableDense(input_dim, activation=None, target_sparsity=0.25, l1_reg=0.001, name='prune_dense_initial')(inputs)
    X_inputs=LeakyReLU(alpha=0.2)(X_inputs)
    ##transformer operations
    tsu_layer_1=TransSingleUnit(input_dim,num_heads,ffn_hidden_dim)
    ##
    X_1=tsu_layer_1(X_inputs)
    for i in range(trans_levels-1):
        tsu_layer_i=TransSingleUnit(input_dim,num_heads,ffn_hidden_dim)
        X_1=tsu_layer_i(X_1)
    Y=ReduceSpatial_v2()(X_1,inputs2)
    Y=Flatten()(Y)
    # Wrap final Dense layer in a Sequential model and prune it
    Y = PrunableDense(num_classes, activation=None, target_sparsity=0.5, l1_reg=0.001,name='prune_dense_final')(Y)
    # Y=Dense(units=num_classes,activation=None)(Y)
    Y = Softmax()(Y)
    model=Model(inputs=[inputs,inputs2], outputs=Y)
    adam = Adam(learning_rate=base_learning_rate)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def total_hsimage_prediction(patch_size,modelhsi):
    count=0
    patch_predictions=np.zeros(gt_temp.shape)
    extra_width=patch_size//2
    ## Padding
    temp_pad=np.pad(temp,((extra_width,extra_width),(extra_width,extra_width),(0,0),(0,0)),mode='edge')
    gt_temp_pad=np.pad(gt_temp,((extra_width,extra_width),(extra_width,extra_width)),mode='edge')
    print('Temp pad shape : ',temp_pad.shape,' GT pad shape : ',gt_temp_pad.shape)
    print('temp shape : ',temp.shape,' GT shape : ',gt_temp.shape)
    ## Patches Extracting
    i_h=temp_pad.shape[0]
    i_w=temp_pad.shape[1]
    ##
    total_samples=int(np.sum(gt_temp>0))
    print('Total samples to predict : ',total_samples)
    label_positions=np.zeros((total_samples,2),dtype=int)
    X_all=np.zeros((total_samples,patch_size,patch_size,temp.shape[-2],temp.shape[-1]))
    sample_count=0

    for hi in range(extra_width,i_h-extra_width):
        for wi in range(extra_width,i_w-extra_width):
            mini_class=gt_temp_pad[hi][wi]
            if(mini_class==0):
                continue
            h_start=hi-extra_width
            w_start=wi-extra_width
            h_end=hi+extra_width
            w_end=wi+extra_width
            #
            mini_patch=temp_pad[h_start:h_end+1,w_start:w_end+1,:,:]
            X_all[sample_count,:,:,:,:]=mini_patch
            #
            label_positions[sample_count][0]=h_start
            label_positions[sample_count][1]=w_start
            sample_count=sample_count+1


    X2_all=cal_SAM_mas_Xformer(X_all)
    X2_all=np.tile(np.expand_dims(X2_all,axis=(-2,-1)),(1,1,1,X_all.shape[-2],X_all.shape[-1]))
    print('Input1 shape: ',X_all.shape)
    print('Input2 shape: ',X2_all.shape)
    #
    Y_pred_hot=modelhsi.predict([X_all,X2_all],batch_size=40,verbose=0)
    Y_pred=hot_to_labels(Y_pred_hot)
    print(f"total sample = {total_samples}, len of pridiction = {len(Y_pred)}")
    for i in range(total_samples):
      x_pos=label_positions[i][0]
      y_pos=label_positions[i][1]
      patch_predictions[x_pos][y_pos]=Y_pred[i].item()
      # if(patch_predictions[x_pos][y_pos]==gt_temp[x_pos][y_pos]):
      #   count=count+1
    print('Correct count : ',int( np.sum(patch_predictions==gt_temp)-np.sum(gt_temp==0) ) )
    return patch_predictions

## Execution Parameters
base_learning_rate=0.0001
total_epochs=1000
batch_size=16
rl_l1_l2=regularizers.L1L2(l1=0,l2=0.0001)
rl_l1=regularizers.L1L2(l1=0.001,l2=0)
p_size=11
dataset_classes=16 # For Salinas 16 and for Pavia Centre 9
trans_fe_levels=25

## Indian Pines - Number of samples for Training and Validation.
Trn_IP=[0,10,71,41,11,24,36,10,23,10,48,122,29,10,63,19,10]
Vld_IP=[0,10,71,41,11,24,36,10,23,5,48,122,29,10,63,19,10]

## Pavia Centre - Number of samples for Training and Validation
# Trn_IP=[0,20,20,20,20,20,20,20,20,20]
# Vld_IP=[0,20,20,20,20,20,20,20,20,20]

## Salinas - Number of samples for Training and Validation
# Trn_IP=[0,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20]
# Vld_IP=[0,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20]




## File loading coding lines, Uncomment and edit according to your file directory
# file_path_temp = '/content/drive/MyDrive/IndianPines/indian_pines_corrected.npy'
# temp = np.load(file_path_temp)
# file_path_gt_temp = '/content/drive/MyDrive/IndianPines/indian_pines_gt.npy'
# gt_temp=np.load(file_path_gt_temp)
# temp_og=np.copy(temp)
# temp=temp.reshape(temp.shape[0],temp.shape[1],20,10)
# temp,pca_obj=PCA_fit_transform(temp,5)
# temp=np.transpose(temp,axes=[0,1,3,2])
# print(temp.shape)

## File loading coding lines, Uncomment and edit according to your file directory
# file_path_temp = '/content/drive/MyDrive/Salinas/salinas_corrected.npy'
# temp = np.load(file_path_temp)
# file_path_gt_temp = '/content/drive/MyDrive/Salinas/salinas_gt.npy'
# gt_temp=np.load(file_path_gt_temp)
# temp_og=np.copy(temp)
# temp=temp[:,:,0:200]
# temp=temp.reshape(temp.shape[0],temp.shape[1],20,10)
# temp,pca_obj=PCA_fit_transform(temp,5)
# temp=np.transpose(temp,axes=[0,1,3,2])
# print(temp.shape)

## File loading coding lines, Uncomment and edit according to your file directory
# file_path_temp = '/content/drive/MyDrive/PaviaCentre/pavia_centre.npy'
# temp = np.load(file_path_temp)
# file_path_gt_temp = '/content/drive/MyDrive/PaviaCentre/pavia_centre_gt.npy'
# gt_temp=np.load(file_path_gt_temp)
# temp_og=np.copy(temp)
# temp=temp[:,:,0:100]
# temp=temp.reshape(temp.shape[0],temp.shape[1],10,10)
# temp,pca_obj=PCA_fit_transform(temp,5)
# temp=np.transpose(temp,axes=[0,1,3,2])
# print(temp.shape)




all_class_matrix=np.zeros((dataset_classes,10))
all_AOK_matrix=np.zeros((3,10))
for rsi in range(10):
    print('-----------------------------------------------------')
    ran_seed=rsi+100
    np.random.seed(ran_seed)
    X_Train,Y_Train,Y_Tr_hot,X_Test,Y_Test,Y_Ts_hot,X_Vld,Y_Vld,Y_Vld_hot,X_og_Tr,X_og_Vld,X_og_Ts=dataset_preparation(p_size)
    X_og_Tr_Ex=np.tile(np.expand_dims(X_og_Tr,axis=(-2,-1)),(1,1,1,X_Train.shape[-2],X_Train.shape[-1]))
    X_og_Vld_Ex=np.tile(np.expand_dims(X_og_Vld,axis=(-2,-1)),(1,1,1,X_Train.shape[-2],X_Train.shape[-1]))
    X_og_Ts_Ex=np.tile(np.expand_dims(X_og_Ts,axis=(-2,-1)),(1,1,1,X_Train.shape[-2],X_Train.shape[-1]))
    print(X_og_Tr_Ex.shape,X_og_Vld_Ex.shape,X_og_Ts_Ex.shape)
    #modelXformer(input_shape,num_heads,trans_levels,num_classes)
    hsicl=modelXformer((None,None,5,10),5,trans_fe_levels,dataset_classes)
    if(rsi==0):
      print(hsicl.summary())
    layers_to_prune = [hsicl.get_layer('prune_dense_initial'),hsicl.get_layer('prune_dense_final')]
    pruning_cb = PruningCallback(layers_to_prune, start_epoch=20, end_epoch=30)
    early_stopping = EarlyStopping(monitor='val_loss',mode='min',patience=10,restore_best_weights=True)
    # reduce_lr_op=ReduceLROnPlateau(monitor="val_loss",factor=0.9,patience=5,verbose=0,min_delta=0.0001,min_lr=0)
    #hsicl.summary()
    history=hsicl.fit([X_Train,X_og_Tr_Ex],Y_Tr_hot,epochs=total_epochs,batch_size=batch_size, verbose=0,
                      validation_data=([X_Vld,X_og_Vld_Ex],Y_Vld_hot),callbacks=[early_stopping,pruning_cb])
    Y_pred_hot=hsicl.predict([X_Test,X_og_Ts_Ex],batch_size=40,verbose=0)
    Y_Pred=hot_to_labels(Y_pred_hot)
    print('Metrics for rand seed: ',ran_seed)
    all_class_matrix[:,rsi:rsi+1],all_AOK_matrix[0,rsi],all_AOK_matrix[1,rsi]=cal_class_accuracies(Y_Test,Y_Pred,dataset_classes)
    kappa = cohen_kappa_score(Y_Test,Y_Pred)
    print("Kappa Coefficient: ", kappa)
    all_AOK_matrix[2,rsi]=kappa
    print('No Epochs', len(history.history['loss']))
    #
    del X_Train,Y_Train,Y_Tr_hot,X_Test,Y_Test,Y_Ts_hot,X_Vld,Y_Vld,Y_Vld_hot,X_og_Tr,X_og_Vld,X_og_Ts
    del X_og_Tr_Ex,X_og_Vld_Ex,X_og_Ts_Ex
    #
    # pp_ip=total_hsimage_prediction(p_size,hsicl)
    # file_path_full_image = '/content/drive/MyDrive/PaviaCentre/CPTNet_Full_Image_PC.npy'
    # np.save(file_path_full_image,pp_ip)
    #
    del hsicl,history,early_stopping,Y_pred_hot,Y_Pred
    # del layers_to_prune,pruning_cb




print('********-----------------------------------------------------------------------------------*********')

print('All Statistics')
for cls in range(dataset_classes):
    print(np.mean(all_class_matrix[cls]))

print('mean of AA ',np.mean(all_AOK_matrix[0]))
print('std of AA ',np.std(all_AOK_matrix[0]))

print('mean of OA',np.mean(all_AOK_matrix[1]))
print('std of OA',np.std(all_AOK_matrix[1]))

print('mean of kappa',np.mean(all_AOK_matrix[2]))
print('std of kappa',np.std(all_AOK_matrix[2]))
