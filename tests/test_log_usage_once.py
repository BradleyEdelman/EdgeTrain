import os
import csv
from datetime import datetime
from unittest.mock import patch, MagicMock
from edgetrain import log_usage_once, sys_resources

def test_log_usage_once(tmpdir):
    # Mock resource usage values
    mock_resources = {
        'cpu_compute_percent': 35.0,
        'cpu_memory_percent': 40.0,
        'gpu_compute_percent': 45.0,
        'gpu_memory_percent': 50.0
    }
    
    # Mock sys_resources to return the mock values
    with patch('edgetrain.sys_resources', return_value=mock_resources):
        # Create a temporary log file
        log_file = os.path.join(tmpdir, 'test_log.csv')

        # Call the function to log usage
        lr = 0.001
        batch_size = 32
        grad_accum = 1
        num_epoch = 5
        log_usage_once(log_file, lr, batch_size, grad_accum, num_epoch)

        # Verify the log file is created
        assert os.path.exists(log_file), "Log file was not created."

        # Read the log file and verify contents
        with open(log_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            # Check if the header is correct
            expected_header = [
                'Timestamp', 'Epoch #', 'CPU Usage (%)', 'CPU RAM (%)',
                'GPU RAM (%)', 'GPU Usage (%)',
                'Batch Size', 'Learning Rate', 'Grad Accum'
            ]
            assert reader.fieldnames == expected_header, "Log file header is incorrect."

            # Check if the log entry contains expected values
            assert len(rows) == 1, "Log file should contain one entry."
            log_entry = rows[0]
            assert log_entry['Epoch #'] == str(num_epoch), "Epoch number mismatch."
            assert log_entry['Batch Size'] == str(batch_size), "Batch size mismatch."
            assert log_entry['Learning Rate'] == str(lr), "Learning rate mismatch."
            assert log_entry['Grad Accum'] == str(grad_accum), "Grad accumulation mismatch."

            # Validate timestamp format
            try:
                datetime.strptime(log_entry['Timestamp'], "%Y-%m-%d %H:%M:%S")
            except ValueError:
                assert False, "Timestamp format is incorrect."

