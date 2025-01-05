import pytest
from scaleml.strategy import set_distributed_strategy

@pytest.mark.parametrize("framework", ["tensorflow", "pytorch", "horovod"])
def test_distributed_strategy(framework):
    strategy = set_distributed_strategy(framework=framework)
    assert strategy is not None, f"{framework} strategy should be initialized"
