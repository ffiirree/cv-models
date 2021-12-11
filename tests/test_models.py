import pytest
import torch
from cvm.models.core import SegmentationModel
from cvm.utils import list_models, create_model


@pytest.mark.parametrize('name', list_models('cvm'))
def test_model_forward(name):
    model = create_model(
        name,
        dropout_rate=0.,
        drop_path_rate=0.,
        num_classes=10,
        cuda=False
    )
    
    model.eval()
    
    inputs = torch.randn((1, 3, 224, 224))
    outputs = model(inputs)
    
    if name in ['unet', 'vae', 'dcgan']:
        ...
    elif isinstance(model, SegmentationModel):
        assert outputs[0].shape == torch.Size([1, 10, 224, 224])
        assert not torch.isnan(outputs[0]).any(), 'Output included NaNs'
    else:
        assert outputs.shape == torch.Size([1, 10])
        assert not torch.isnan(outputs).any(), 'Output included NaNs'
