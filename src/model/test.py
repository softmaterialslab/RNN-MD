
from model.commen_libs import *
from model.LSTM import LSTM_Model

class RNN_Test(tf.Module):
	def __init__(self, experiment=None):
		super(RNN_Test, self).__init__()
		"""
		Class for the Training and testing LSTM model
		"""
		self.experiment=experiment
		self.load_config()

		self.model = LSTM_Model(self.input_shape, self.dropout_rate, self.lstmUnits1, self.lstmUnits2, self.output_shape)

		self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)


	def load_config(self):
		try:
			with open("config/"+self.experiment+'.yaml') as f:
				dictionary = yaml.load(f, Loader=yaml.FullLoader)
				#print(dictionary)
				self.input_features=dictionary['data']['input_features']
				self.window_size=dictionary['data']['window_size']
				self.input_shape= (self.window_size, self.input_features) 
				self.output_shape=dictionary['data']['output_shape']
				self.reduction_factor =dictionary['data']['reduction_factor']
				self.train_test_split = dictionary['data']['train_test_split']
				self.batchSize=dictionary['data']['batchSize']
				self.dataset_dir = dictionary['data']['dataset_dir']
				self.dataset_name = dictionary['data']['dataset_name']
				self.selection_index = dictionary['data']['selection_index']
				self.selected_training_count = dictionary['data']['selected_training_count']
				

				self.lstmUnits1=dictionary['model']['lstmUnits1']
				self.lstmUnits2=dictionary['model']['lstmUnits2']
				self.saved_models_dir=dictionary['model']['saved_models_dir']
				self.saved_models_name=dictionary['model']['saved_models_name']
				
				self.dropout_rate=dictionary['train']['dropout_rate']
				self.learning_rate=dictionary['train']['learning_rate']
				self.epochs=dictionary['train']['epochs']

		except  Exception as e:
			sys.stderr.write('Error occured during the configuration load')
			sys.stderr.write(str(e))
			sys.exit(1)

	def load_data(self):
		"""
		Class for the Training and testing LSTM model
		"""
		with open(self.dataset_dir+self.dataset_name, 'rb') as handle:
		    (self.input_list, self.all_data, self.training_indexes, self.testing_indexes) = pickle.load(handle)

		print("Total # of simulation:" +str(len(self.input_list)))
		print(self.all_data.shape)
		print(len(self.training_indexes))
		print(len(self.testing_indexes))


		fig=plt.figure(figsize=(20, 6))
		plt.title('Training data')

		
		self.input_data = []
		self.output = []

		for sim_ in self.training_indexes[0:self.selected_training_count]:
		  selected_data = self.all_data[sim_][::self.reduction_factor,self.selection_index]  # 6 referes to actual cos solution
		  plt.plot(self.all_data[sim_][::self.reduction_factor,0], selected_data, label=self.input_list[sim_], linewidth=1, markersize=3)
		  for i in range(self.window_size, selected_data.shape[0]):
		        self.input_data.append(selected_data[(i-self.window_size):i])
		        self.output.append(selected_data[i])

		plt.xlabel('time')
		plt.ylabel('Position')
		plt.legend()
		#plt.ylim(-2.5, 10)
		plt.xlim(0, 40)

		self.input_data = np.array(self.input_data)
		self.output = np.array(self.output)
		print(self.input_data.shape)
		print(self.output.shape)


		self.input_data_suff, self.output_suff  = shuffle(self.input_data, self.output)

		
		self.train_test_split_ = int(self.input_data_suff.shape[0]*self.train_test_split)

		self.x_train = self.input_data_suff[0:self.train_test_split_].reshape(-1,self.window_size,self.input_features)
		self.x_test = self.input_data_suff[self.train_test_split_:].reshape(-1,self.window_size,self.input_features)
		self.y_train = self.output_suff[0:self.train_test_split_].reshape(-1,self.output_shape)
		self.y_test = self.output_suff[self.train_test_split_:].reshape(-1,self.output_shape)

		print("input: ", self.input_data_suff.shape)
		print("Output", self.output_suff.shape)
		print("Train input: ", self.x_train.shape)
		print("Train Output", self.y_train.shape)
		print("Test input: ", self.x_test.shape)
		print("Test Output", self.y_test.shape)

		self.dataset = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
		self.dataset = self.dataset.shuffle(20).batch(self.batchSize)

	def load_many_particle_data(self):
		"""
		Class for the Training and testing LSTM model
		"""
		with open(self.dataset_dir+self.dataset_name, 'rb') as handle:
		    (self.input_list, self.all_data, self.training_indexes, self.testing_indexes) = pickle.load(handle)

		print("Total # of simulation:" +str(len(self.input_list)))
		print(self.all_data.shape)
		print(len(self.training_indexes))
		print(len(self.testing_indexes))

		self.input_data = []
		self.output = []


		for sim_ in self.training_indexes[0:self.selected_training_count]:
		  selected_data = self.all_data[sim_][::self.reduction_factor]  # 6 referes to actual cos solution
		  for i in range(self.window_size, selected_data.shape[0]):
		        self.input_data.append(selected_data[(i-self.window_size):i])
		        self.output.append(selected_data[i])

		
		self.input_data = np.array(self.input_data)
		self.output = np.array(self.output)
		print(self.input_data.shape)
		print(self.output.shape)


		self.input_data_suff, self.output_suff  = shuffle(self.input_data, self.output)

		
		self.train_test_split_ = int(self.input_data_suff.shape[0]*self.train_test_split)

		self.x_train = self.input_data_suff[0:self.train_test_split_].reshape(-1,self.window_size,self.input_features)
		self.x_test = self.input_data_suff[self.train_test_split_:].reshape(-1,self.window_size,self.input_features)
		self.y_train = self.output_suff[0:self.train_test_split_].reshape(-1,self.output_shape)
		self.y_test = self.output_suff[self.train_test_split_:].reshape(-1,self.output_shape)

		print("input: ", self.input_data_suff.shape)
		print("Output", self.output_suff.shape)
		print("Train input: ", self.x_train.shape)
		print("Train Output", self.y_train.shape)
		print("Test input: ", self.x_test.shape)
		print("Test Output", self.y_test.shape)

		self.dataset = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
		self.dataset = self.dataset.shuffle(20).batch(self.batchSize)


	@tf.function
	def loss_func(self, targets, logits):
	    #print(targets.shape)
	    #print(logits.shape)
	    #loss = tf.reduce_mean(tf.square(logits - targets))
	    #reduce_sum
	    loss = tf.math.reduce_mean(tf.keras.losses.mean_squared_error(targets, logits))
	    return loss

	@tf.function
	def train_step(self, source_seq, target_seq_out):
		""" Execute one training step (forward pass + backward pass)
		Args:
			source_seq: source sequences
			target_seq_in: input target sequences (x0,x1,....xn)
			target_seq_out: output target sequences (xn+1)
		Returns:
			The loss value of the current pass
		"""
		with tf.GradientTape() as tape:

			model_output = self.model(source_seq)
			#print(model_output.shape)

			loss = self.loss_func(target_seq_out, model_output)

		variables = self.model.trainable_variables
		gradients = tape.gradient(loss, variables)
		self.optimizer.apply_gradients(zip(gradients, variables))

		return loss

	
	def predict(self, test_source=None):
	    """ Predict the output sentence for a given input sentence
	    Args:
	        test_source_text: input sentence 
	    
	    Returns:
	    print("Test input: ", x_test.shape)
	    print("Test Output", y_test.shape)   
	    """
	    target_test = None
	    loss=-1.0
	    if test_source is None:
	        test_source = self.x_test
	        target_test = self.y_test

	    en_output = self.model(test_source, training=False)

	    if target_test is not None:
	      loss = self.loss_func(target_test, en_output)
	      #print("Test data loss: {}".format(loss.numpy()))

	      return loss.numpy(), en_output
	    else:
	      return en_output


	def train(self, dataset=None):
		self.train_loss = []
		self.test_loss = []

		starttime = time.time()
		for e in range(self.epochs):
		  loss_value=0.0
		  train_batch_loss=[]
		  for batch, (source_seq, target_seq_out) in enumerate(self.dataset.take(-1)):
		    #print(source_seq.shape)
		    #print(target_seq_out.shape)
		    loss = self.train_step(source_seq, target_seq_out)
		    #break
		    #print(loss)
		    loss_value = loss.numpy()
		    train_batch_loss.append(loss_value)
		    #if batch % 1000 == 0:
		    #  print('Epoch {} Batch {} Traing batch Loss {} Elapsed time {:.2f}s'.format(e + 1, batch, loss_value, time.time() - starttime)) 
		  self.train_loss.append(np.mean(train_batch_loss))
		  try:
		    #pass
		    loss_value, _ = self.predict()
		    self.test_loss.append(loss_value)
		    print('Epoch {}, Training Loss {}, Test Loss {}, Elapsed time {:.2f}s'.format(e + 1, self.train_loss[-1], self.test_loss[-1], time.time() - starttime)) 
		    starttime = time.time()
		  except Exception as e:
		    print(e)
		    continue


	def save_model(self):
		
		self.model.summary()
		# This is 8 time frames
		#model.evaluate(x_test, y_test)
		# Save the model as a hdf5 file
		# if you get folder creation error, manually create the folder
		tf.keras.models.save_model(model=self.model,filepath=self.saved_models_dir+self.saved_models_name)

		fig, ax = plt.subplots(1,1)
		ax.plot(self.train_loss, color='b', label="Training loss")
		ax.plot(self.test_loss, color='r', label="validation loss",axes =ax)
		plt.yscale('log')
		legend = ax.legend(loc='best', shadow=True)

		#ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
		#ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
		#legend = ax[1].legend(loc='best', shadow=True)


	def load_model(self):

		self.model = tf.keras.models.load_model(filepath=self.saved_models_dir+self.saved_models_name, compile=True)

		self.model.summary()


	def simulate_new(self, testing_index=1):
		#sim_ =training_indexes[0]
		self.sim_ =self.testing_indexes[testing_index]

		selected_data = self.all_data[self.sim_][::self.reduction_factor,self.selection_index]

		self.actual_output = []
		self.predicted_output = []

		for i in range(self.window_size, selected_data.shape[0]):
		  self.predicted_output.append(self.predict(selected_data[(i-self.window_size):i].reshape(-1, self.window_size, self.input_features)))
		  self.actual_output.append(selected_data[i])

		self.actual_output = np.array(self.actual_output)
		self.predicted_output = np.array(self.predicted_output).reshape(-1)

		# This is to check continous RNN prediction
		self.Only_RNN_predicted_output = []

		temp__ = selected_data[0:self.window_size]
		temp__ = np.append(temp__, self.predicted_output, axis=0)

		for i in range(self.window_size, selected_data.shape[0]):
		  self.Only_RNN_predicted_output.append(self.predict(temp__[(i-self.window_size):i].reshape(-1, self.window_size, self.input_features)))

		self.Only_RNN_predicted_output = np.array(self.Only_RNN_predicted_output).reshape(-1)


		print(self.actual_output.shape)
		print(self.predicted_output.shape)
		print(self.Only_RNN_predicted_output.shape)
		#print(predicted_output)


	def simulate_new_many_particle(self, testing_index=1):
		#sim_ =training_indexes[0]
		self.sim_ =self.testing_indexes[testing_index]

		selected_data = self.all_data[self.sim_][::self.reduction_factor]

		self.actual_output = []
		self.predicted_output = []

		for i in range(self.window_size, selected_data.shape[0]):
		  self.predicted_output.append(self.predict(selected_data[(i-self.window_size):i].reshape(-1, self.window_size, self.input_features)))
		  self.actual_output.append(selected_data[i])

		self.actual_output = np.array(self.actual_output)
		self.predicted_output = np.array(self.predicted_output).reshape(-1,selected_data.shape[1], selected_data.shape[2])

		print("Actual data shape: "+str(self.actual_output.shape))
		print("Predicted data shape: "+str(self.predicted_output.shape))
		#print(predicted_output)

		self.make_movie(file_name='actual.lammpstrj', data=self.actual_output)
		self.make_movie(file_name='predicted.lammpstrj', data=self.predicted_output)

  
	def make_movie(self, file_name = '', data =None):

		lx=6.0
		ly=6.0
		lz=6.0

		#path = os.getcwd()
		#print(path)
		temp_dir = 'temp_data/'

		for num in range(data.shape[0]):
			if num==0:
				outdump = open(temp_dir+file_name, "w")
			else:
				outdump = open(temp_dir+file_name, "a")
	    
			outdump.write("ITEM: TIMESTEP\n")
			outdump.write("{}\n".format(num - 1))
			outdump.write("ITEM: NUMBER OF ATOMS\n")
			outdump.write("{}\n".format(data.shape[1]))
			outdump.write("ITEM: BOX BOUNDS\n")
			outdump.write("{}\t{}\n".format(-0.5*lx, 0.5*lx))
			outdump.write("{}\t{}\n".format(-0.5*lx, 0.5*ly))
			outdump.write("{}\t{}\n".format(-0.5*lz, 0.5*lz))
			outdump.write("ITEM: ATOMS index type x y z v\n")

			for j in range(data.shape[1]):
				outdump.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(j+1, "1", data[num][j][0], data[num][j][1], data[num][j][2], 0)) 
			outdump.close()
      