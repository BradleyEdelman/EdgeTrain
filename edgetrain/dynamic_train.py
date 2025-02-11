import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow import keras

from edgetrain import (
    adjust_training_parameters,
    compute_scores,
    create_model_tf,
    define_priorities,
    log_usage_once,
)


def dynamic_train(
    train_dataset,
    epochs=10,
    batch_size=32,
    lr=1e-3,
    pruning=0.2,
    log_file="resource_log.csv",
    dynamic_adjustments=True,
):
    """
    Train the model with optional dynamic resource adjustment.

    Parameters:
    - train_dataset (dict): The training dataset containing 'images' and 'labels'.
    - epochs (int): Number of epochs to train the model.
    - batch_size (int): The base batch size to use.
    - lr (float): The initial learning rate.
    - pruning (float): Initial pruning ratio (for dynamic adjustment).
    - log_file (str): Path to the log file where resource usage is saved.
    - dynamic_adjustments (bool): Flag to enable/disable dynamic adjustments.

    Returns:
    - final_model (tf.keras.Model): The trained and stripped model.
    - history_list (list): A list of training history for each epoch.
    """

    # Log initial resource usage
    normalized_scores = {"memory_score": 0, "accuracy_score": 0}
    priority_value = {"batch_size": 0, "learning_rate": 0}
    log_usage_once(
        log_file,
        pruning,
        batch_size,
        lr,
        normalized_scores,
        priority_value,
        num_epoch=0,
        resources=None,
    )

    # Create MirroredStrategy for distributed training
    strategy = tf.distribute.MirroredStrategy()

    # Initialize training variables
    history_list = []
    prev_accuracy = 0.0

    # Prepare training data
    train_images, train_labels = train_dataset["images"], train_dataset["labels"]

    # Create model within scope and apply initial pruning
    with strategy.scope():
        base_model = create_model_tf(input_shape=train_images[0].shape)
        optimizer = keras.optimizers.Adam(learning_rate=lr)

        pruning_schedule = tfmot.sparsity.keras.ConstantSparsity(pruning, begin_step=0)
        model = tfmot.sparsity.keras.prune_low_magnitude(
            base_model, pruning_schedule=pruning_schedule
        )
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    # Training model one epoch at a time
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # Add pruning update callback
        callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]

        history = model.fit(
            train_images,
            train_labels,
            batch_size=batch_size,
            epochs=1,
            callbacks=callbacks,
        )

        # Save training history
        history_list.append(history.history)

        # Update "current" accuracy
        curr_accuracy = history.history["accuracy"][-1]

        # If dynamic adjustments are enabled
        if dynamic_adjustments:
            # Compute scores & priorities
            normalized_scores = compute_scores(prev_accuracy, curr_accuracy)
            priority_value = define_priorities(normalized_scores)

            # Adjust highest priority parameter
            adjusted_batch_size, adjusted_lr = adjust_training_parameters(
                priority_values=priority_value,
                batch_size=batch_size,
                lr=lr,
                accuracy_score=curr_accuracy,
            )

            batch_size = adjusted_batch_size
            lr = adjusted_lr

            print(
                f"Adjusted parameters for next epoch: batch_size={batch_size}, pruning_ratio={pruning}, learning_rate={lr}"
            )

        # Log resource usage
        log_usage_once(
            log_file,
            pruning,
            batch_size,
            lr,
            normalized_scores,
            priority_value,
            num_epoch=epoch + 1,
            resources=None,
        )

        # Update previous accuracy
        prev_accuracy = curr_accuracy

    # Strip pruning for final model deployment
    final_model = tfmot.sparsity.keras.strip_pruning(model)
    print("Pruning stripped. Model ready for deployment.")

    return final_model, history_list
