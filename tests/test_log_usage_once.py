import csv
import os
from datetime import datetime

from edgetrain import compute_scores, define_priorities, log_usage_once


def test_log_usage_once(tmpdir):
    # Mock resource usage
    mock_resources = {
        "num_gpus": 0,
        "cpu_compute_percent": 30.0,
        "cpu_memory_percent": 40.0,
        "gpu_compute_percent": 45.0,
        "gpu_memory_percent": 50.0,
    }

    # Create a temporary log file
    log_file = os.path.join(tmpdir, "test_log.csv")

    # Call the function to log usage
    lr = 0.001
    batch_size = 32
    pruning = 0.2
    num_epoch = 5
    prev_accuracy = 0.8
    curr_accuracy = 0.6

    # Calculate performance and resource usage scores
    normalized_scores = compute_scores(
        prev_accuracy, curr_accuracy, score_ranges=None, resources=mock_resources
    )
    priority_value = define_priorities(normalized_scores)

    log_usage_once(
        log_file,
        pruning,
        batch_size,
        lr,
        normalized_scores,
        priority_value,
        num_epoch,
        resources=mock_resources,
    )

    # Verify the log file is created
    assert os.path.exists(log_file), "Log file was not created."

    # Read the log file and verify contents
    with open(log_file, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

        # Check if the header is correct
        expected_header = [
            "Timestamp",
            "Epoch #",
            "CPU Usage (%)",
            "CPU RAM (%)",
            "GPU RAM (%)",
            "GPU Usage (%)",
            "Mem Score",
            "Acc Score",
            "Priority Batch Size",
            "Priority Learning Rate",
            "Pruning",
            "Batch Size",
            "Learning Rate",
        ]
        assert reader.fieldnames == expected_header, "Log file header is incorrect."

        # Check if the log entry contains expected values
        assert len(rows) == 1, "Log file should contain one entry."
        log_entry = rows[0]
        assert log_entry["Epoch #"] == str(num_epoch), "Epoch number mismatch."
        assert log_entry["Mem Score"] == str(
            normalized_scores.get("memory_score")
        ), "Mem score mismatch."
        assert log_entry["Acc Score"] == str(
            normalized_scores.get("accuracy_score")
        ), "Acc score mismatch."
        assert log_entry["Priority Batch Size"] == str(
            priority_value.get("batch_size")
        ), "Priority batch size mismatch."
        assert log_entry["Priority Learning Rate"] == str(
            priority_value.get("learning_rate")
        ), "Priority learning rate mismatch."
        assert log_entry["Pruning"] == str(pruning), "Pruning ratio mismatch."
        assert log_entry["Batch Size"] == str(batch_size), "Batch size mismatch."
        assert log_entry["Learning Rate"] == str(lr), "Learning rate mismatch."

        # Validate timestamp format
        try:
            datetime.strptime(log_entry["Timestamp"], "%Y-%m-%d %H:%M:%S")
        except ValueError:
            assert False, "Timestamp format is incorrect."
