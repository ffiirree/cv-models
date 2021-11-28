import argparse
from torchinfo import summary
from cvm.utils import create_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--model', '-m', type=str)

    args = parser.parse_args()

    model = create_model(args.model, cuda=False)

    summary(
        model,
        input_size=(1, 3, 224, 224),
        col_names=("output_size", "num_params", 'mult_adds')
    )
