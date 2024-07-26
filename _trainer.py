
import tensorflow as tf
import tensorflow_transform as tft
from tensorflow.keras import layers
import tensorflow_hub as hub
from tfx.components.trainer.fn_args_utils import FnArgs

_LABEL_KEY = 'rating'
_FEATURE = 'review'

def _transformed_name(key):
    return key + '_xf'

def _gzip_reader_fn(filenames):

    '''Loads compressed data'''
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def _input_fn(file_pattern,
             tf_transform_output,
             num_epochs,
             batch_size=64)->tf.data.Dataset:

    # Get post_transform feature spec
    transform_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy())

    # create batches of data
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=_gzip_reader_fn,
        num_epochs=num_epochs,
        label_key = _transformed_name(_LABEL_KEY))
    return dataset



embed = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4")
def model_builder():

    rate = 0.2

    inputs = tf.keras.Input(shape=(1,), name=_transformed_name('review'), dtype=tf.string)
    reshaped_narrative = tf.reshape(inputs, [-1])
    x = embed(reshaped_narrative)
    x = tf.keras.layers.Reshape((1,512), input_shape=(1,512))(x)
    x = layers.Dense(64, activation='elu', kernel_initializer='glorot_uniform')(x)

    attn_output = layers.MultiHeadAttention(num_heads=2, key_dim=64)(x, x, x)
    attn_output = layers.Dropout(rate)(attn_output)

    out1 = layers.LayerNormalization(epsilon=1e-7)(x + attn_output)
    ffn_output = layers.Dense(64, activation="elu", kernel_initializer="glorot_uniform")(out1)
    ffn_output = layers.Dense(64, kernel_initializer='glorot_uniform')(ffn_output)
    ffn_output = layers.Dropout(rate)(ffn_output)

    x = layers.LayerNormalization(epsilon=1e-7)(out1 + ffn_output)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(rate)(x)
    x = layers.Dense(32, activation="elu", kernel_initializer="glorot_uniform")(x)
    x = layers.Dropout(rate)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)


    model = tf.keras.Model(inputs=inputs, outputs = outputs)

    model.compile(
        loss = 'binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(0.01),
        metrics=[tf.keras.metrics.BinaryAccuracy()]

    )

    # print(model)
    model.summary()
    return model


def _get_serve_tf_examples_fn(model, tf_transform_output):

    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):

        feature_spec = tf_transform_output.raw_feature_spec()

        feature_spec.pop("rating")

        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)

        transformed_features = model.tft_layer(parsed_features)

        # get predictions using the transformed features
        return model(transformed_features)

    return serve_tf_examples_fn

def run_fn(fn_args: FnArgs) -> None:

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir = fn_args.model_run_dir, update_freq='batch'
    )

    es = tf.keras.callbacks.EarlyStopping(monitor='val_binary_accuracy', mode='max', verbose=1, patience=10)
    mc = tf.keras.callbacks.ModelCheckpoint(fn_args.serving_model_dir, monitor='val_binary_accuracy', mode='max', verbose=1, save_best_only=True)


    # Load the transform output
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    # Create batches of data
    train_set = _input_fn(fn_args.train_files, tf_transform_output, 10)
    val_set = _input_fn(fn_args.eval_files, tf_transform_output, 10)


    # Build the model
    model = model_builder()


    # Train the model
    model.fit(x = train_set,
             validation_data = val_set,
             callbacks = [tensorboard_callback, es, mc],
               steps_per_epoch = 1000,
             validation_steps= 1000,
             epochs=1)
    signatures = {
        'serving_default':
        _get_serve_tf_examples_fn(model,
                                 tf_transform_output).get_concrete_function(
                                    tf.TensorSpec(
                                    shape=[None],
                                    dtype=tf.string,
                                    name='examples'))
    }
    model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
