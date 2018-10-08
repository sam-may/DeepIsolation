import tensorflow as tf

def parallel(charged_pf_timestep, n_charged_pf_features, photon_pf_timestep, n_photon_pf_features, neutralHad_pf_timestep, n_neutralHad_pf_features, n_global_features):
  input_charged_pf = tf.keras.layers.Input(shape=(charged_pf_timestep, n_charged_pf_features), name = 'charged_pf')
  input_photon_pf = tf.keras.layers.Input(shape=(photon_pf_timestep, n_photon_pf_features), name = 'photon_pf')
  input_neutralHad_pf = tf.keras.layers.Input(shape=(neutralHad_pf_timestep, n_neutralHad_pf_features), name = 'neutralHad_pf')
  input_global = tf.keras.layers.Input(shape=(n_global_features,), name = 'global')

  # Convolutional layers for pf cands
  dropout_rate_1 = 0.1
  maxnorm = 3
  conv_charged_pf = tf.keras.layers.Convolution1D(32, 1, kernel_initializer = 'lecun_uniform',  activation = 'relu', name = 'conv_charged_pf_1')(input_charged_pf)
  conv_charged_pf = tf.keras.layers.Dropout(dropout_rate_1, name = 'cpf_dropout_1')(conv_charged_pf)
  conv_charged_pf = tf.keras.layers.Convolution1D(16, 1, kernel_initializer = 'lecun_uniform',  activation = 'relu', name = 'conv_charged_pf_2')(conv_charged_pf)
  conv_charged_pf = tf.keras.layers.Dropout(dropout_rate_1, name = 'cpf_dropout_2')(conv_charged_pf)
  conv_charged_pf = tf.keras.layers.Convolution1D(16, 1, kernel_initializer = 'lecun_uniform',  activation = 'relu', name = 'conv_charged_pf_3')(conv_charged_pf)
  conv_charged_pf = tf.keras.layers.Dropout(dropout_rate_1, name = 'cpf_dropout_3')(conv_charged_pf)
  conv_charged_pf = tf.keras.layers.Convolution1D(4, 1, kernel_initializer = 'lecun_uniform',  activation = 'relu', name = 'conv_charged_pf_4')(conv_charged_pf)

  conv_photon_pf = tf.keras.layers.Convolution1D(24, 1, kernel_initializer = 'lecun_uniform',  activation = 'relu', name = 'conv_photon_pf_1')(input_photon_pf)
  conv_photon_pf = tf.keras.layers.Dropout(dropout_rate_1, name = 'ppf_dropout_1')(conv_photon_pf)
  conv_photon_pf = tf.keras.layers.Convolution1D(12, 1, kernel_initializer = 'lecun_uniform',  activation = 'relu', name = 'conv_photon_pf_2')(conv_photon_pf)
  conv_photon_pf = tf.keras.layers.Dropout(dropout_rate_1, name = 'ppf_dropout_2')(conv_photon_pf)
  conv_photon_pf = tf.keras.layers.Convolution1D(3, 1, kernel_initializer = 'lecun_uniform',  activation = 'relu', name = 'conv_photon_pf_4')(conv_photon_pf)

  conv_neutralHad_pf = tf.keras.layers.Convolution1D(24, 1, kernel_initializer = 'lecun_uniform',  activation = 'relu', name = 'conv_neutralHad_pf_1')(input_neutralHad_pf)
  conv_neutralHad_pf = tf.keras.layers.Dropout(dropout_rate_1, name = 'npf_dropout_1')(conv_neutralHad_pf)
  conv_neutralHad_pf = tf.keras.layers.Convolution1D(12, 1, kernel_initializer = 'lecun_uniform',  activation = 'relu', name = 'conv_neutralHad_pf_2')(conv_neutralHad_pf)
  conv_neutralHad_pf = tf.keras.layers.Dropout(dropout_rate_1, name = 'npf_dropout_2')(conv_neutralHad_pf)
  conv_neutralHad_pf = tf.keras.layers.Convolution1D(3, 1, kernel_initializer = 'lecun_uniform',  activation = 'relu', name = 'conv_neutralHad_pf_4')(conv_neutralHad_pf)

  # LSTMs for pf cands
  batch_momentum = 0.6
  go_backwards = True
  print("Go backwards = " + str(go_backwards))

  lstm_charged_pf = tf.keras.layers.LSTM(150, implementation = 2, name ='lstm_charged_pf_1', go_backwards = go_backwards)(conv_charged_pf)
  lstm_charged_pf = tf.keras.layers.Dropout(dropout_rate_1, name = 'lstm_charged_pf_dropout')(lstm_charged_pf)

  lstm_photon_pf = tf.keras.layers.LSTM(100, implementation = 2, name = 'lstm_photon_pf_1', go_backwards = go_backwards)(conv_photon_pf)
  lstm_photon_pf = tf.keras.layers.Dropout(dropout_rate_1, name = 'lstm_photon_pf_dropout')(lstm_photon_pf)

  lstm_neutralHad_pf = tf.keras.layers.LSTM(50, implementation = 2, name = 'lstm_neutralHad_pf_1', go_backwards = go_backwards)(conv_neutralHad_pf)
  lstm_neutralHad_pf = tf.keras.layers.Dropout(dropout_rate_1, name = 'lstm_neutralHad_pf_dropout')(lstm_neutralHad_pf)

  # MLP to combine LSTM outputs with global features
  dropout_rate_2 = 0.2
  cand_features = tf.keras.layers.concatenate([lstm_charged_pf, lstm_photon_pf, lstm_neutralHad_pf])
  deep_layer = tf.keras.layers.Dense(200, activation = 'relu', kernel_initializer = 'lecun_uniform',  name = 'mlp_1')(cand_features)
  deep_layer = tf.keras.layers.Dropout(dropout_rate_2, name = 'mlp_dropout_1')(deep_layer)
  deep_layer = tf.keras.layers.Dense(100, activation = 'relu', kernel_initializer = 'lecun_uniform',  name = 'mlp_2')(deep_layer)
  deep_layer = tf.keras.layers.Dropout(dropout_rate_2, name = 'mlp_dropout_2')(deep_layer)
  deep_layer = tf.keras.layers.Dense(100, activation = 'relu', kernel_initializer = 'lecun_uniform',  name = 'mlp_3')(deep_layer)
  deep_layer = tf.keras.layers.Dropout(dropout_rate_2, name = 'mlp_dropout_3')(deep_layer)
  deep_layer = tf.keras.layers.Dense(100, activation = 'relu', kernel_initializer = 'lecun_uniform',  name = 'mlp_4')(deep_layer)
  deep_layer = tf.keras.layers.Dropout(dropout_rate_2, name = 'mlp_dropout_4')(deep_layer)
  deep_layer = tf.keras.layers.Dense(100, activation = 'relu', kernel_initializer = 'lecun_uniform',  name = 'mlp_7')(deep_layer)

  dropout_rate_3 = 0.1
  deep_layer_global = tf.keras.layers.Dense(64, activation = 'relu', kernel_initializer = 'lecun_uniform',  name = 'mlp_g_1')(input_global)
  deep_layer_global = tf.keras.layers.Dropout(dropout_rate_3, name = 'mlp_dropout_global_1')(deep_layer_global)
  deep_layer_global = tf.keras.layers.Dense(64, activation = 'relu', kernel_initializer = 'lecun_uniform',  name = 'mlp_g_2')(deep_layer_global)
  deep_layer_global = tf.keras.layers.Dropout(dropout_rate_3, name = 'mlp_dropout_global_2')(deep_layer_global)
  deep_layer_global = tf.keras.layers.Dense(64, activation = 'relu', kernel_initializer = 'lecun_uniform',  name = 'mlp_g_3')(deep_layer_global)
  deep_layer_global = tf.keras.layers.Dropout(dropout_rate_3, name = 'mlp_dropout_global_3')(deep_layer_global)
  deep_layer_global = tf.keras.layers.Dense(64, activation = 'relu', kernel_initializer = 'lecun_uniform',  name = 'mlp_g_4')(deep_layer_global)

  dropout_rate_4 = 0.2
  merged_features = tf.keras.layers.concatenate([deep_layer, deep_layer_global])
  deep_layer_merged = tf.keras.layers.Dense(100, activation = 'relu', kernel_initializer = 'lecun_uniform',  name = 'mlp_m_1')(merged_features)
  deep_layer_merged = tf.keras.layers.Dropout(dropout_rate_4, name = 'mlp_dropout_m_1')(deep_layer_merged)
  deep_layer_merged = tf.keras.layers.Dense(100, activation = 'relu', kernel_initializer = 'lecun_uniform',  name = 'mlp_m_2')(deep_layer_merged)
  deep_layer_merged = tf.keras.layers.Dropout(dropout_rate_4, name = 'mlp_dropout_m_2')(deep_layer_merged)
  deep_layer_merged = tf.keras.layers.Dense(100, activation = 'relu', kernel_initializer = 'lecun_uniform',  name = 'mlp_m_3')(deep_layer_merged)
  deep_layer_merged = tf.keras.layers.Dropout(dropout_rate_4, name = 'mlp_dropout_m_3')(deep_layer_merged)
  output = tf.keras.layers.Dense(1, activation = 'sigmoid', kernel_initializer = 'lecun_uniform', name = 'output')(deep_layer_merged)

  model = tf.keras.models.Model(inputs = [input_charged_pf, input_photon_pf, input_neutralHad_pf, input_global], outputs = [output])
  return model
