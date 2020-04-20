
from model.commen_libs import *

class LSTM_Model(tf.keras.Model):
    """ Class for the LSTM model
        Args:
        input_shape: 

    """
    def __init__(self, input_shape_lstm, dropout_rate, lstmUnits1, lstmUnits2, output_shape_dense):
        super(LSTM_Model, self).__init__()
        self.seed = None
        self.init_model = 'fan_in'
        self.init_distribution = 'truncated_normal'
        self.initializer = tf.keras.initializers.VarianceScaling(scale=2.0, mode=self.init_model, distribution=self.init_distribution, seed=self.seed)

        ## LSTM LAYER 01
        self.input_shape_lstm = input_shape_lstm
        self.lstm1_size = lstmUnits1
        self.dropout_rate = dropout_rate
        self.lstm1 = tf.keras.layers.LSTM(self.lstm1_size, kernel_initializer=self.initializer, input_shape=self.input_shape_lstm, return_sequences=True)
        self.batchNormal1 = None
        self.batchNormal1 = tf.keras.layers.BatchNormalization()
        self.activation1 = tf.keras.layers.Activation('relu')
        self.dropout1 = tf.keras.layers.Dropout(rate=self.dropout_rate)

        ## LSTM LAYER 02
        self.lstm2_size = lstmUnits2
        self.lstm2 = tf.keras.layers.LSTM(self.lstm2_size, kernel_initializer=self.initializer)
        self.batchNormal2 = None
        self.batchNormal2 = tf.keras.layers.BatchNormalization()
        self.activation2 = tf.keras.layers.Activation('relu')
        self.dropout2 = tf.keras.layers.Dropout(rate=self.dropout_rate)

        ## Dense LAYER
        self.output_shape_dense = output_shape_dense
        self.dense1 = tf.keras.layers.Dense(self.output_shape_dense, activation=None, kernel_initializer=self.initializer)


    def call(self, sequence, training=True):
        """ Forward pass for the LSTM
            Args:
              sequence: source input sequences
              training: whether training or not (for Dropout)

            Returns:
              The output of the LSTM networkk (batch_size, input features)

        """

        lstm_out = self.lstm1(sequence)
        #print(lstm_out.shape)
        if self.batchNormal1:
            lstm_out = self.batchNormal1(lstm_out)
        lstm_out = self.activation1(lstm_out)
        if training:
            lstm_out = self.dropout1(lstm_out)

        lstm_out = self.lstm2(lstm_out)
        if self.batchNormal2:
            lstm_out = self.batchNormal2(lstm_out)
        lstm_out = self.activation2(lstm_out)
        if training:
            lstm_out = self.dropout2(lstm_out)    

        fc_output = self.dense1(lstm_out)

        return fc_output
