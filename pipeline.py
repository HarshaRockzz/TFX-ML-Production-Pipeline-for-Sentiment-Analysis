import os
import glob
import socket
import tensorflow as tf
from tfx.components import CsvExampleGen, StatisticsGen, SchemaGen, ExampleValidator, Transform, Tuner, Trainer, Pusher, Evaluator, ResolverNode
from tfx.dsl.components.base import executor_spec
from tfx.dsl.experimental.latest_blessed_model_strategy import LatestBlessedModelStrategy
from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext
from tfx.proto import example_gen_pb2, tuner_pb2, trainer_pb2, pusher_pb2, eval_config_pb2
from tfx.types.standard_artifacts import Model, ModelBlessing
from pyngrok import ngrok
from tensorboard import program

# Initialize the TFX interactive context
context = InteractiveContext()

# Define dataset paths
datasets = [
    '/content/drive/MyDrive/1_movies_per_genre',
    '/content/drive/MyDrive/2_reviews_per_movie_raw',
    '/content/drive/MyDrive/Horror.csv',
    '/content/drive/MyDrive/Animation.csv'
]

current_dataset_index = 0

def ensure_consistent_headers(csv_files):
    # Add implementation to ensure headers are consistent across all CSV files
    pass

# Function to find an available port
def find_available_port(start_port=6006, max_attempts=100):
    port = start_port
    while port < start_port + max_attempts:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', port)) != 0:
                return port
        port += 1
    raise RuntimeError('No available ports found')

# Placeholder for DriftDetection component
class DriftDetection:
    # Implement DriftDetection logic here
    pass

# Loop through datasets to run the pipeline components and perform drift detection
while current_dataset_index < len(datasets):
    dataset_path = datasets[current_dataset_index]
    csv_files = glob.glob(os.path.join(dataset_path, '*.csv'))
    ensure_consistent_headers(csv_files)

    # Initialize ExampleGen component
    example_gen = CsvExampleGen(
        input_base=os.path.dirname(dataset_path),
        input_config=example_gen_pb2.Input(splits=[
            example_gen_pb2.Input.Split(name='train', pattern=os.path.basename(dataset_path))
        ]),
        output_config=example_gen_pb2.Output(
            split_config=example_gen_pb2.SplitConfig(splits=[
                example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=2),
            ])
        )
    )

    # Initialize other TFX components
    statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])
    schema_gen = SchemaGen(statistics=statistics_gen.outputs['statistics'])
    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema']
    )
    
    # Transform component
    transform = Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file='/content/_transform.py'  # Update with your actual path
    )

    # Tuner component
    tuner = Tuner(
        module_file='/content/_tuner.py',  # Update with your actual path
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        train_args=trainer_pb2.TrainArgs(num_steps=1000),
        eval_args=trainer_pb2.EvalArgs(num_steps=500)
    )

    # Trainer component
    trainer = Trainer(
        module_file='/content/_trainer.py',  # Update with your actual path
        custom_executor_spec=executor_spec.ExecutorClassSpec(trainer.executor.GenericExecutor),
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        hyperparameters=tuner.outputs['best_hyperparameters'],
        train_args=trainer_pb2.TrainArgs(num_steps=1000),
        eval_args=trainer_pb2.EvalArgs(num_steps=500)
    )

    # Model Resolver component
    model_resolver = ResolverNode(
        instance_name='latest_blessed_model_resolver',
        strategy_class=LatestBlessedModelStrategy,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing)
    )

    # Evaluator component
    eval_config = eval_config_pb2.EvalConfig(
        model_specs=[eval_config_pb2.ModelSpec(label_key='label')],
        slicing_specs=[eval_config_pb2.SlicingSpec()],
        metrics_specs=[
            eval_config_pb2.MetricsSpec(
                metrics=[
                    eval_config_pb2.MetricConfig(class_name='SparseCategoricalAccuracy'),
                    eval_config_pb2.MetricConfig(class_name='ExampleCount')
                ]
            )
        ]
    )

    evaluator = Evaluator(
        examples=example_gen.outputs['examples'],
        model=trainer.outputs['model'],
        eval_config=eval_config
    )

    # Pusher component
    pusher = Pusher(
        model=trainer.outputs['model'],
        model_blessing=evaluator.outputs['blessing'],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(base_directory='/content/pipeline/serving_model')  # Update with your actual path
        )
    )

    # Run the components
    context.run(example_gen)
    context.run(statistics_gen)
    context.run(schema_gen)
    context.run(example_validator)
    context.run(transform)
    context.run(tuner)
    context.run(trainer)
    context.run(model_resolver)
    context.run(evaluator)
    context.run(pusher)

    # Print ExampleValidator output artifact URI
    print("ExampleValidator anomalies URI:", example_validator.outputs['anomalies'].get()[0].uri)

    # Implement DriftDetection logic here
    drift_detection = DriftDetection()
    # context.run(drift_detection)
    # drift_detected = context.show(drift_detection.outputs['drift_detected'])

    # if drift_detected == 'true':
    #     print(f"Drift detected in {dataset_path}. Retraining on next dataset.")
    # else:
    #     print("No drift detected. Proceeding to the next dataset.")

    current_dataset_index += 1

# TensorBoard and ngrok setup
parent_log_dir = '/content/pipeline/logs'  # Update with your actual logs path
available_port = find_available_port()
tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', parent_log_dir, '--port', str(available_port)])
url = tb.launch()
public_url = ngrok.connect(addr=str(available_port), proto='http')
print(f'Combined TensorBoard URL: {public_url}')
