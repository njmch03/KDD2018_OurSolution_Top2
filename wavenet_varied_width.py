import tensorflow as tf
import pandas as pd
import numpy as np
import os.path
import os, sys
import argparse
from datetime import datetime

from helper import erfinv_rank_transform, log100p, expm100
from data_frame import DataFrameKDD

class DataReader(object):
    def __init__(self, data_dir, x_encode_len=24*14):
        data_cols = [
            'aq',
            'aq_norm',
            'aq_isnan',
            'meo_his_humidity',
            'meo_his_pressure',
            'meo_his_temperature',
            'meo_his_wind_direction',
            'meo_his_wind_speed',
            'meo_pred_humidity',
            'meo_pred_pressure',
            'meo_pred_temperature',
            'meo_pred_wind_direction',
            'meo_pred_wind_speed',
            'dow_cos',
            'dow_sin',
            'hod_cos',
            'hod_sin',
            'station_id_elemnt',
            'city_id',
            'elemnt_id',
            'lbs_aq_self'
        ]
        data = [np.load(os.path.join(data_dir, '{}.npy'.format(i))) for i in data_cols]

        self.test_df = DataFrameKDD(columns=data_cols, data=data)
        self.train_df, self.val_df = self.test_df.train_test_split_KDD(fake_ind=1)

        self.x_encode_len = x_encode_len

        print('train size', len(self.train_df))
        print('val size', len(self.val_df))
        print('test size', len(self.test_df))

    def train_batch_generator(self, batch_size, train_start_point=24*30):
        return self.batch_generator(
            batch_size=batch_size,
            df=self.train_df,
            shuffle=True,
            num_epochs=1000000,
            train_start_point=train_start_point,
            x_encode_len=self.x_encode_len,
            mode='trn'
        )

    def val_batch_generator(self, batch_size):
        return self.batch_generator(
            batch_size=batch_size,
            df=self.val_df,
            shuffle=True,
            num_epochs=1000000,
            x_encode_len=self.x_encode_len,
            mode='val'
        )

    def test_batch_generator(self, batch_size, x_encode_end_backshift=0):
        return self.batch_generator(
            batch_size=batch_size,
            df=self.test_df,
            shuffle=True,
            num_epochs=1,
            x_encode_len=self.x_encode_len,
            x_encode_end_backshift=x_encode_end_backshift,
            mode='test'
        )

    def batch_generator(self, batch_size, df, shuffle=True, num_epochs=10000, train_start_point=24*90
              , x_encode_len=24*14, x_encode_end_backshift=0, mode='trn'):

        batch_gen = df.batch_generator(
            batch_size=batch_size,
            shuffle=shuffle,
            num_epochs=num_epochs,
            allow_smaller_final_batch=(mode=='test')
        )

        for batch in batch_gen:
            num_decode_steps = 50
            full_seq_len = batch['aq'].shape[1]
            max_encode_length = x_encode_len

            x_encode = np.zeros([len(batch), max_encode_length])
            is_nan_encode = np.zeros([len(batch), max_encode_length])

            y_decode = np.zeros([len(batch), num_decode_steps])
            is_nan_decode = np.zeros([len(batch), num_decode_steps])

            encode_len = np.zeros([len(batch)])
            decode_len = np.zeros([len(batch)])

            meo_his_humidity          =  np.zeros([len(batch), max_encode_length])
            meo_his_pressure          =  np.zeros([len(batch), max_encode_length])
            meo_his_temperature       =  np.zeros([len(batch), max_encode_length])
            meo_his_wind_direction    =  np.zeros([len(batch), max_encode_length])
            meo_his_wind_speed        =  np.zeros([len(batch), max_encode_length])

            meo_pred_humidity         =  np.zeros([len(batch), num_decode_steps])
            meo_pred_pressure         =  np.zeros([len(batch), num_decode_steps])
            meo_pred_temperature      =  np.zeros([len(batch), num_decode_steps])
            meo_pred_wind_direction   =  np.zeros([len(batch), num_decode_steps])
            meo_pred_wind_speed       =  np.zeros([len(batch), num_decode_steps])

            aq_norm =  np.zeros([len(batch), max_encode_length])
            lbs_aq_self =  np.zeros([len(batch), max_encode_length])

            dow_cos_x =  np.zeros([len(batch), max_encode_length])
            dow_sin_x =  np.zeros([len(batch), max_encode_length])
            hod_cos_x =  np.zeros([len(batch), max_encode_length])
            hod_sin_x =  np.zeros([len(batch), max_encode_length])

            dow_cos_pred =  np.zeros([len(batch), num_decode_steps])
            dow_sin_pred =  np.zeros([len(batch), num_decode_steps])
            hod_cos_pred =  np.zeros([len(batch), num_decode_steps])
            hod_sin_pred =  np.zeros([len(batch), num_decode_steps])

            ### training, validation, test
            for i, (seq, taq_norm, nan_seq, tmeo_his_humidity, tmeo_his_pressure, tmeo_his_temperature
                    , tmeo_his_wind_direction, tmeo_his_wind_speed
                    , tmeo_pred_humidity, tmeo_pred_pressure, tmeo_pred_temperature, tmeo_pred_wind_direction, tmeo_pred_wind_speed
                    , tdow_cos, tdow_sin, thod_cos, thod_sin, tlbs_aq_self
                   ) in enumerate(zip(batch['aq']
                                        , batch['aq_norm']
                                        , batch['aq_isnan']
                                        , batch['meo_his_humidity']
                                        , batch['meo_his_pressure']
                                        , batch['meo_his_temperature']
                                        , batch['meo_his_wind_direction']
                                        , batch['meo_his_wind_speed']
                                        , batch['meo_pred_humidity']
                                        , batch['meo_pred_pressure']
                                        , batch['meo_pred_temperature']
                                        , batch['meo_pred_wind_direction']
                                        , batch['meo_pred_wind_speed']
                                        , batch['dow_cos']
                                        , batch['dow_sin']
                                        , batch['hod_cos']
                                        , batch['hod_sin']
                                        , batch['lbs_aq_self']
                                       )):

                ############# random split  #############
                if mode=='test':
                    x_encode_end = full_seq_len - 1 - x_encode_end_backshift
                elif mode=='val':
                    back_days = np.random.randint(3, 30)
                    x_encode_end = full_seq_len - 24*back_days -1
                elif mode=='trn':
                    x_encode_end = np.random.randint(train_start_point, full_seq_len-24*31 - 1)

                x_encode_start = x_encode_end - x_encode_len

                x_encode[i, :x_encode_len] = seq[x_encode_start: x_encode_end]
                is_nan_encode[i, :x_encode_len] = nan_seq[x_encode_start: x_encode_end]

                encode_len[i] = x_encode_len
                decode_len[i] = num_decode_steps

                ############# moe his
                meo_his_humidity[i, :x_encode_len]        = tmeo_his_humidity[x_encode_start: x_encode_end]
                meo_his_pressure[i, :x_encode_len]        = tmeo_his_pressure[x_encode_start: x_encode_end]
                meo_his_temperature[i, :x_encode_len]     = tmeo_his_temperature[x_encode_start: x_encode_end]
                meo_his_wind_direction[i, :x_encode_len]  = tmeo_his_wind_direction[x_encode_start: x_encode_end]
                meo_his_wind_speed[i, :x_encode_len]      = tmeo_his_wind_speed[x_encode_start: x_encode_end]

                aq_norm[i, :x_encode_len]          = taq_norm[x_encode_start: x_encode_end]
                lbs_aq_self[i, :x_encode_len]      = tlbs_aq_self[x_encode_start: x_encode_end]

                ############# moe pred
                meo_pred_humidity[i, :]       = tmeo_pred_humidity[x_encode_end: x_encode_end + num_decode_steps]
                meo_pred_pressure[i, :]       = tmeo_pred_pressure[x_encode_end: x_encode_end + num_decode_steps]
                meo_pred_temperature[i, :]    = tmeo_pred_temperature[x_encode_end: x_encode_end + num_decode_steps]
                meo_pred_wind_direction[i, :] = tmeo_pred_wind_direction[x_encode_end: x_encode_end + num_decode_steps]
                meo_pred_wind_speed[i, :]     = tmeo_pred_wind_speed[x_encode_end: x_encode_end + num_decode_steps]

                dow_cos_x[i, :]     = tdow_cos[x_encode_start: x_encode_end]
                dow_sin_x[i, :]     = tdow_sin[x_encode_start: x_encode_end]
                hod_cos_x[i, :]     = thod_cos[x_encode_start: x_encode_end]
                hod_sin_x[i, :]     = thod_sin[x_encode_start: x_encode_end]

                dow_cos_pred[i, :]     = tdow_cos[x_encode_end: x_encode_end + num_decode_steps]
                dow_sin_pred[i, :]     = tdow_sin[x_encode_end: x_encode_end + num_decode_steps]
                hod_cos_pred[i, :]     = thod_cos[x_encode_end: x_encode_end + num_decode_steps]
                hod_sin_pred[i, :]     = thod_sin[x_encode_end: x_encode_end + num_decode_steps]

                if not mode=='test':
                    y_decode[i, :] = seq[x_encode_end: x_encode_end + num_decode_steps]
                    is_nan_decode[i, :] = nan_seq[x_encode_end: x_encode_end + num_decode_steps]

            x_encode[x_encode<0] = 0
            x_encode[x_encode>300] = 300

            y_decode[y_decode<0] = 0
            y_decode[y_decode>300] = 300

            batch['x_encode'] = x_encode
            batch['aq_norm'] = aq_norm
            batch['encode_len'] = encode_len
            batch['y_decode'] = y_decode
            batch['decode_len'] = decode_len
            batch['is_nan_encode'] = is_nan_encode
            batch['is_nan_decode'] = is_nan_decode

            batch['meo_his_humidity']        =  meo_his_humidity
            batch['meo_his_pressure']        =  meo_his_pressure
            batch['meo_his_temperature']     =  meo_his_temperature
            batch['meo_his_wind_direction']  =  meo_his_wind_direction
            batch['meo_his_wind_speed']      =  meo_his_wind_speed

            batch['lbs_aq_self'] = lbs_aq_self

            batch['meo_pred_humidity']       =  meo_pred_humidity
            batch['meo_pred_pressure']       =  meo_pred_pressure
            batch['meo_pred_temperature']    =  meo_pred_temperature
            batch['meo_pred_wind_direction'] =  meo_pred_wind_direction
            batch['meo_pred_wind_speed']     =  meo_pred_wind_speed

            batch['dow_cos_pred'] =  dow_cos_pred
            batch['dow_sin_pred'] =  dow_sin_pred
            batch['hod_cos_pred'] =  hod_cos_pred
            batch['hod_sin_pred'] =  hod_sin_pred

            batch['dow_cos_x'] =  dow_cos_x
            batch['dow_sin_x'] =  dow_sin_x
            batch['hod_cos_x'] =  hod_cos_x
            batch['hod_sin_x'] =  hod_sin_x

            yield batch

class wavenet(Object):

    def __init__(
        self,
        residual_channels=32,
        skip_channels=32,
        dilations=[2**i for i in range(3)],
        filter_widths=[2 for i in range(3)],
        num_decode_steps=32,
        regularization_continuity=0.01,
        **kwargs
        ):
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.dilations = dilations
        self.filter_widths = filter_widths
        self.num_decode_steps = num_decode_steps
        self.regularization_continuity = regularization_continuity
        super(cnn, self).__init__(**kwargs)


    def future_reduce(self):
        def reduce_self(x):
            return tf.concat([tf.reduce_mean(x, axis=1, keepdims=True),\
                               tf.reduce_max( x, axis=1, keepdims=True),\
                               tf.reduce_min( x, axis=1, keepdims=True)], axis=1)
        return tf.concat([ reduce_self(self.meo_pred_humidity       )\
                     , reduce_self(self.meo_pred_pressure       )\
                     , reduce_self(self.meo_pred_temperature    )\
                     , reduce_self(self.meo_pred_wind_direction )\
                     , reduce_self(self.meo_pred_wind_speed     )   ], axis=1)

    def transform(self, x):
        return tf.log(x + 100) - self.log_x_encode_mean

    def inverse_transform(self, x):
        return tf.exp(x + self.log_x_encode_mean) - 100

    def get_input_sequences(self):
        self.x_encode = tf.placeholder(tf.float32, [None, None])
        self.encode_len = tf.placeholder(tf.int32, [None])
        self.y_decode = tf.placeholder(tf.float32, [None, self.num_decode_steps])
        self.decode_len = tf.placeholder(tf.int32, [None])
        self.is_nan_encode = tf.placeholder(tf.float32, [None, None])
        self.is_nan_decode = tf.placeholder(tf.float32, [None, self.num_decode_steps])

        self.station_id_elemnt = tf.placeholder(tf.string, [None])

        self.city_id          = tf.placeholder(tf.int32, [None])
        self.elemnt_id        = tf.placeholder(tf.int32, [None])

        self.meo_his_humidity            = tf.placeholder(tf.float32, [None, None])
        self.meo_his_pressure            = tf.placeholder(tf.float32, [None, None])
        self.meo_his_temperature         = tf.placeholder(tf.float32, [None, None])
        self.meo_his_wind_direction      = tf.placeholder(tf.float32, [None, None])
        self.meo_his_wind_speed          = tf.placeholder(tf.float32, [None, None])

        self.aq_norm = tf.placeholder(tf.float32, [None, None])
        self.lbs_aq_self = tf.placeholder(tf.float32, [None, None])

        self.meo_pred_humidity           = tf.placeholder(tf.float32, [None, self.num_decode_steps])
        self.meo_pred_pressure           = tf.placeholder(tf.float32, [None, self.num_decode_steps])
        self.meo_pred_temperature        = tf.placeholder(tf.float32, [None, self.num_decode_steps])
        self.meo_pred_wind_direction     = tf.placeholder(tf.float32, [None, self.num_decode_steps])
        self.meo_pred_wind_speed         = tf.placeholder(tf.float32, [None, self.num_decode_steps])

        self.dow_cos_x = tf.placeholder(tf.float32, [None, None])
        self.dow_sin_x = tf.placeholder(tf.float32, [None, None])
        self.hod_cos_x = tf.placeholder(tf.float32, [None, None])
        self.hod_sin_x = tf.placeholder(tf.float32, [None, None])

        self.dow_cos_pred = tf.placeholder(tf.float32, [None, self.num_decode_steps])
        self.dow_sin_pred = tf.placeholder(tf.float32, [None, self.num_decode_steps])
        self.hod_cos_pred = tf.placeholder(tf.float32, [None, self.num_decode_steps])
        self.hod_sin_pred = tf.placeholder(tf.float32, [None, self.num_decode_steps])

        self.keep_prob = tf.placeholder(tf.float32)
        self.is_training = tf.placeholder(tf.bool)

        ### log100p, minus mean
        self.log_x_encode_mean = sequence_mean(tf.log(self.x_encode + 100), self.encode_len)

        self.log_x_encode_mean_3d = tf.reduce_mean(tf.log(self.x_encode[:,-24*3:] + 100), axis=1, keepdims=True)
        self.log_x_encode_mean_1d = tf.reduce_mean(tf.log(self.x_encode[:,-24:] + 100), axis=1, keepdims=True)
        self.log_x_encode_max_1d  = tf.reduce_max(tf.log(self.x_encode[:,-24:] + 100), axis=1, keepdims=True)


        self.log_x_encode = self.transform(self.x_encode)
        self.x = tf.expand_dims(self.log_x_encode, 2)

        x_length = tf.cast(tf.shape(self.x_encode)[1], tf.float32)
        encode_idx = tf.tile(tf.expand_dims(tf.range(x_length, dtype=tf.float32)/x_length*2-1, 0)\
                          , (tf.shape(self.x_encode)[0], 1))
        decode_idx = tf.tile(tf.expand_dims(tf.range(self.num_decode_steps, dtype=tf.float32)/self.num_decode_steps*2-1, 0)\
                                    , (tf.shape(self.y_decode)[0], 1))
        self.encode_features = tf.concat([
            tf.tile(tf.expand_dims(tf.one_hot(self.city_id, 2), 1), (1, tf.shape(self.x_encode)[1], 1))
            , tf.tile(tf.expand_dims(tf.one_hot(self.elemnt_id, 3), 1), (1, tf.shape(self.x_encode)[1], 1))
            , tf.expand_dims(self.is_nan_encode, 2)
            , tf.expand_dims(tf.cast(tf.equal(self.x_encode, 0.0), tf.float32), 2)
            , tf.expand_dims(self.aq_norm, 2)
            , tf.expand_dims(self.lbs_aq_self, 2)
            , tf.expand_dims(self.dow_cos_x, 2)
            , tf.expand_dims(self.dow_sin_x, 2)
            , tf.expand_dims(self.hod_cos_x, 2)
            , tf.expand_dims(self.hod_sin_x, 2)
            , tf.expand_dims(self.meo_his_humidity, 2)
            , tf.expand_dims(self.meo_his_pressure, 2)
            , tf.expand_dims(self.meo_his_temperature, 2)
            , tf.expand_dims(self.meo_his_wind_direction, 2)
            , tf.expand_dims(self.meo_his_wind_speed, 2)
            , tf.tile(tf.expand_dims(self.future_reduce(), 1), (1, tf.shape(self.x_encode)[1], 1))
            , tf.expand_dims(encode_idx, 2)
        ], axis=2)

        self.decode_features = tf.concat([
            tf.tile(tf.expand_dims(tf.one_hot(self.city_id, 2), 1), (1, self.num_decode_steps, 1))
            , tf.tile(tf.expand_dims(tf.one_hot(self.elemnt_id, 3), 1), (1, self.num_decode_steps, 1))
            , tf.expand_dims(self.dow_cos_pred, 2)
            , tf.expand_dims(self.dow_sin_pred, 2)
            , tf.expand_dims(self.hod_cos_pred, 2)
            , tf.expand_dims(self.hod_sin_pred, 2)
            , tf.expand_dims(self.meo_pred_humidity, 2)
            , tf.expand_dims(self.meo_pred_pressure, 2)
            , tf.expand_dims(self.meo_pred_temperature, 2)
            , tf.expand_dims(self.meo_pred_wind_direction, 2)
            , tf.expand_dims(self.meo_pred_wind_speed, 2)
            , tf.tile(tf.expand_dims(self.log_x_encode_mean, 1), (1, self.num_decode_steps, 1))
            , tf.tile(tf.expand_dims(self.log_x_encode_mean_1d, 1), (1, self.num_decode_steps, 1))
            , tf.tile(tf.expand_dims(self.log_x_encode_mean_3d, 1), (1, self.num_decode_steps, 1))
            , tf.tile(tf.expand_dims(self.log_x_encode_max_1d, 1), (1, self.num_decode_steps, 1))
            , tf.tile(tf.expand_dims(self.future_reduce(), 1), (1, self.num_decode_steps, 1))
            , tf.expand_dims(decode_idx, 2)
            ], axis=2)

        return self.x

    def encode(self, x, features):
        x = tf.concat([x, features], axis=2)

        ## dense layers
        inputs = time_distributed_dense_layer(
            inputs=x,
            output_units=self.residual_channels,
            activation=tf.nn.tanh,
            dropout=self.keep_prob,
            scope='x-proj-encode'
        )

        skip_outputs = []
        conv_inputs = [inputs]
        for i, (dilation, filter_width) in enumerate(zip(self.dilations, self.filter_widths)):
            dilated_conv = time_convolution_layer(
                inputs=inputs,
                output_units=2*self.residual_channels,
                convolution_width=filter_width,
                causal=True,
                dilation_rate=[dilation],
                scope='dilated-conv-encode-{}'.format(i)
            )
            conv_filter, conv_gate = tf.split(dilated_conv, 2, axis=2)
            dilated_conv = tf.nn.tanh(conv_filter)*tf.nn.sigmoid(conv_gate)

            outputs = time_distributed_dense_layer(
                inputs=dilated_conv,
                output_units=self.skip_channels + self.residual_channels,
                scope='dilated-conv-proj-encode-{}'.format(i)
            )
            skips, residuals = tf.split(outputs, [self.skip_channels, self.residual_channels], axis=2)

            inputs += residuals
            conv_inputs.append(inputs)
            skip_outputs.append(skips)

        skip_outputs = tf.nn.selu(tf.concat(skip_outputs, axis=2))
        h = time_distributed_dense_layer(skip_outputs, 32, scope='dense-encode-1', activation=tf.nn.selu)
        y_hat = time_distributed_dense_layer(h, 1, scope='dense-encode-2')

        return y_hat, conv_inputs[:-1]

    def initialize_decode_params(self, x, features):  ## parameter shape
        x = tf.concat([x, features], axis=2)

        inputs = time_distributed_dense_layer(
            inputs=x,
            output_units=self.residual_channels,
            activation=tf.nn.tanh,
            dropout=self.keep_prob,
            scope='x-proj-decode'
            )

        skip_outputs = []
        conv_inputs = [inputs]
        for i, (dilation, filter_width) in enumerate(zip(self.dilations, self.filter_widths)):
            dilated_conv = time_convolution_layer(
                inputs=inputs,
                output_units=2*self.residual_channels,
                convolution_width=filter_width,
                causal=True,
                dilation_rate=[dilation],
                scope='dilated-conv-decode-{}'.format(i)
            )
            conv_filter, conv_gate = tf.split(dilated_conv, 2, axis=2)
            dilated_conv = tf.nn.tanh(conv_filter)*tf.nn.sigmoid(conv_gate)

            outputs = time_distributed_dense_layer(
                inputs=dilated_conv,
                output_units=self.skip_channels + self.residual_channels,
                scope='dilated-conv-proj-decode-{}'.format(i)
            )
            skips, residuals = tf.split(outputs, [self.skip_channels, self.residual_channels], axis=2)

            inputs += residuals
            conv_inputs.append(inputs)
            skip_outputs.append(skips)

        skip_outputs = tf.nn.selu(tf.concat(skip_outputs, axis=2))
        h = time_distributed_dense_layer(skip_outputs, 32, scope='dense-decode-1', activation=tf.nn.selu)
        y_hat = time_distributed_dense_layer(h, 1, scope='dense-decode-2')
        return y_hat

    def decode(self, x, conv_inputs, features):
        batch_size = tf.shape(x)[0]

        # initialize state tensor arrays
        state_queues = []
        for i, (conv_input, dilation, width) in enumerate(zip(conv_inputs, self.dilations, self.filter_widths)):
            dilation_width = dilation * (width-1)
            batch_idx = tf.range(batch_size)
            batch_idx = tf.tile(tf.expand_dims(batch_idx, 1), (1, dilation_width))
            batch_idx = tf.reshape(batch_idx, [-1])

            queue_begin_time = self.encode_len - dilation_width - 1
            temporal_idx = tf.expand_dims(queue_begin_time, 1) + tf.expand_dims(tf.range(dilation_width), 0)
            temporal_idx = tf.reshape(temporal_idx, [-1])

            idx = tf.stack([batch_idx, temporal_idx], axis=1)
            slices = tf.reshape(tf.gather_nd(conv_input, idx), (batch_size, dilation_width, shape_dim(conv_input, 2)))

            layer_ta = tf.TensorArray(dtype=tf.float32, size=dilation_width + self.num_decode_steps, clear_after_read = False)
            layer_ta = layer_ta.unstack(tf.transpose(slices, (1, 0, 2)))
            state_queues.append(layer_ta)

        # initialize feature tensor array
        features_ta = tf.TensorArray(dtype=tf.float32, size=self.num_decode_steps)

        ### cnn
        features_cnn = time_convolution_layer(features, shape_dim(features,2), 5, causal=False)
        features_ta = features_ta.unstack(tf.transpose(features_cnn, (1, 0, 2)))

        # initialize output tensor array
        emit_ta = tf.TensorArray(size=self.num_decode_steps, dtype=tf.float32)

        # initialize other loop vars
        elements_finished = 0 >= self.decode_len
        time = tf.constant(0, dtype=tf.int32)

        # get initial x input
        current_idx = tf.stack([tf.range(tf.shape(self.encode_len)[0]), self.encode_len - 1], axis=1)
        initial_input = tf.gather_nd(x, current_idx)

        def loop_fn(time, current_input, queues):
            current_features = tf.nn.dropout(features_ta.read(time), self.keep_prob)    ######## current features
            current_input = tf.concat([current_input, current_features], axis=1)

            with tf.variable_scope('x-proj-decode', reuse=True):
                w_x_proj = tf.get_variable('weights')
                b_x_proj = tf.get_variable('biases')
                x_proj = tf.nn.tanh(tf.matmul(current_input, w_x_proj) + b_x_proj)

            skip_outputs, updated_queues = [], []
            for i, (conv_input, queue, dilation, width) in enumerate(zip(conv_inputs, queues, self.dilations, self.filter_widths)):

                with tf.variable_scope('dilated-conv-decode-{}'.format(i), reuse=True):
                    w_conv = tf.get_variable('weights'.format(i))
                    b_conv = tf.get_variable('biases'.format(i))
                    dilated_conv = tf.matmul(x_proj, w_conv[0, :, :]) + b_conv
                    for di in range(width-1):
                        state = queue.read(time + dilation*di)                          ### state time
                        dilated_conv =  dilated_conv + tf.matmul(state, w_conv[di+1, :, :])

                conv_filter, conv_gate = tf.split(dilated_conv, 2, axis=1)
                dilated_conv = tf.nn.tanh(conv_filter)*tf.nn.sigmoid(conv_gate)

                with tf.variable_scope('dilated-conv-proj-decode-{}'.format(i), reuse=True):
                    w_proj = tf.get_variable('weights'.format(i))
                    b_proj = tf.get_variable('biases'.format(i))
                    concat_outputs = tf.matmul(dilated_conv, w_proj) + b_proj
                skips, residuals = tf.split(concat_outputs, [self.skip_channels, self.residual_channels], axis=1)

                x_proj += residuals
                skip_outputs.append(skips)

                dilation_width = dilation * (width-1)
                updated_queues.append(queue.write(dilation_width + time, x_proj))

            skip_outputs = tf.nn.selu(tf.concat(skip_outputs, axis=1))
            with tf.variable_scope('dense-decode-1', reuse=True):
                w_h = tf.get_variable('weights')
                b_h = tf.get_variable('biases')
                h = tf.nn.selu(tf.matmul(skip_outputs, w_h) + b_h)

            with tf.variable_scope('dense-decode-2', reuse=True):
                w_y = tf.get_variable('weights')
                b_y = tf.get_variable('biases')
                y_hat = tf.matmul(h, w_y) + b_y

            elements_finished = (time >= self.decode_len)
            finished = tf.reduce_all(elements_finished)

            next_input = tf.cond(
                finished,
                lambda: tf.zeros([batch_size, 1], dtype=tf.float32),
                lambda: y_hat
            )
            next_elements_finished = (time >= self.decode_len - 1)

            return (next_elements_finished, next_input, updated_queues)

        def condition(unused_time, elements_finished, *_):
            return tf.logical_not(tf.reduce_all(elements_finished))

        def body(time, elements_finished, emit_ta, last_output, *state_queues):
            (next_finished, emit_output, state_queues) = loop_fn(time, last_output, state_queues)

            emit = tf.where(elements_finished, tf.zeros_like(emit_output), emit_output)
            emit_ta = emit_ta.write(time, emit)

            elements_finished = tf.logical_or(elements_finished, next_finished)
            return [time + 1, elements_finished, emit_ta, emit_output] + list(state_queues)

        returned = tf.while_loop(
            cond=condition,
            body=body,
            loop_vars=[time, elements_finished, emit_ta, initial_input] + state_queues)

        outputs_ta = returned[2]
        y_hat = tf.transpose(outputs_ta.stack(), (1, 0, 2))
        return y_hat

    def calculate_loss(self):
        x = self.get_input_sequences()

        y_hat_encode, conv_inputs = self.encode(x, features=self.encode_features)
        self.initialize_decode_params(x, features=self.decode_features)
        y_hat_decode = self.decode(y_hat_encode, conv_inputs, features=self.decode_features)
        y_hat_decode = self.inverse_transform(tf.squeeze(y_hat_decode, 2))

        self.labels = self.y_decode
        self.preds = tf.clip_by_value(y_hat_decode, 0, 300)

        weight = tf.cast(tf.ones_like(self.city_id), tf.float32)
        self.loss = sequence_smape(self.labels, self.preds, self.decode_len, self.is_nan_decode, weight)

        cont_loss = tf.abs(tf.reduce_mean(self.preds[:,1:] - self.preds[:,:-1])) / tf.reduce_mean(tf.abs(self.preds))
        self.loss_trn = self.loss + self.regularization_continuity * cont_loss

        self.prediction_tensors = {
            'priors': self.x_encode,
            'labels': self.labels,
            'preds': self.preds,
            'station_id_elemnt': self.station_id_elemnt,
        }

        return self.loss, self.loss_trn


if __name__ == '__main__':
    data_dir = './data/'
    base_dir = './output/'

    dr = DataReader(data_dir=os.path.join(data_dir, 'wavenet/'))

    nn = wavenet(
        reader=dr,
        log_dir=os.path.join(base_dir, 'logs'),
        checkpoint_dir=os.path.join(base_dir, 'checkpoints'),
        prediction_dir=os.path.join(base_dir, 'predictions'),
        optimizer='adam',
        learning_rate=.003,
        batch_size=64,
        num_training_steps=200000,
        early_stopping_steps=1000,
        warm_start_init_step=0,
        regularization_constant=0.00001,
        regularization_continuity=0.05,
        keep_prob=0.9,
        enable_parameter_averaging=False,
        num_restarts=2,
        min_steps_to_checkpoint=500,
        log_interval=10,
        num_validation_batches=1,
        grad_clip=20,
        residual_channels=32,
        skip_channels=32,
        dilations    =[1, 4, 8, 24, 48],
        filter_widths=[4, 2, 3, 2,  2 ],
        num_decode_steps=50
    )
    nn.fit()
    nn.restore()
    nn.predict()

