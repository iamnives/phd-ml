from models.classification.alex_net import get_model
from models.classification.simple_net import simple_model


class TestSimpleNet:
    def test_compile_net(self):
        simple_model()


class TestAlexNet:
    def test_compile_net(self):
        get_model()
