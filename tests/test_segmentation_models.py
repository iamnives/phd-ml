from models.segmentation.u_net import get_dual_input_siamese_model, get_model, get_siamese_model


class TestUNet:
    def test_dual_input_siamese(self):
        get_dual_input_siamese_model()

    def test_original_net(self):
        get_model()

    def test_siamese_model(self):
        get_siamese_model()
