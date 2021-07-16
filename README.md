# Models

| Network           |Year  | From        | Acc@1    | Acc@5  | Params(M)| MACs(M) |
| --                |--    | --          | --:      | --:    | --:    | --:       |
|AlexNet            | 2012 | torchvision | 56.522 | 79.066 |  61.1008 |   714.692 |
|VGG-11             | 2014 | torchvision | 69.020 | 88.628 | 132.8633 |  7616.566 |
|VGG-11(BN)         | 2014 | torchvision | 70.370 | 89.810 | 132.8688 |  7631.418 |
|VGG-13             | 2014 | torchvision | 69.928 | 89.246 | 133.0478 | 11320.759 |
|VGG-13(BN)         | 2014 | torchvision | 71.586 | 90.374 | 133.0537 | 11345.245 |
|VGG-16             | 2014 | torchvision | 71.592 | 90.382 | 138.3575 | 15483.862 |
|VGG-16(BN)         | 2014 | torchvision | 73.360 | 91.516 | 138.3660 | 15510.957 |
|VGG-19             | 2014 | torchvision | 72.376 | 90.876 | 143.6672 | 19646.965 |
|VGG-19(BN)         | 2014 | torchvision | 74.218 | 91.842 | 143.6782 | 19676.669 |
|ResNet-18          | 2015 | torchvision | 69.758 | 89.078 |  11.6895 |  1819.066 |
|ResNet-34          | 2015 | torchvision | 73.314 | 91.420 |  21.7977 |  3671.263 |
|ResNet-50          | 2015 | torchvision | 76.130 | 92.862 |  25.5570 |  4111.515 |
|ResNet-101         | 2015 | torchvision | 77.374 | 93.546 |  44.5492 |  7833.972 |
|ResNet-152         | 2015 | torchvision | 78.312 | 94.046 |  60.1928 | 11558.837 |
|ResNeXt-50-32x4d   | 2016 | torchvision | 77.618 | 93.698 |  25.0289 |  4259.383 |
|ResNeXt-101-32x8d  | 2016 | torchvision | 79.312 | 94.526 |  88.7913 | 16476.537 |
|Wide ResNet-50-2   | 2016 | torchvision | 78.468 | 94.086 |  68.8832 | 11426.925 |
|Wide ResNet-101-2  | 2016 | torchvision | 78.848 | 94.284 | 126.8867 | 22795.602 |
|SqueezeNet 1.0     | 2016 | torchvision | 58.092 | 80.420 |   1.2484 |   823.441 |
|SqueezeNet 1.1     | 2016 | torchvision | 58.178 | 80.624 |   1.2355 |   351.911 |
|Densenet-121       | 2017 | torchvision | 74.434 | 91.972 |   7.9789 |  2865.672 |
|Densenet-169       | 2017 | torchvision | 75.600 | 92.806 |  14.1495 |  3398.071 |
|Densenet-201       | 2017 | torchvision | 76.896 | 93.370 |  20.0139 |  4340.971 |
|Densenet-161       | 2017 | torchvision | 77.138 | 93.560 |  28.6810 |  7787.013 |
|GoogleNet          | 2014 | torchvision | 69.778 | 89.530 |   6.6249 |  1504.880 |
|Inception v3       | 2017 | torchvision | 77.294 | 93.450 |  23.8346 |  2847.271 |
|MobileNet          | 2017 | Paper       | 70.600 |        |   4.2000 |   569.000 |
|MobileNet V2       | 2018 | torchvision | 71.878 | 90.286 |   3.5049 |   314.130 |
|ShuffleNet 0.5x g=3| 2017 | Paper       | 56.800 |        |          |           |
|ShuffleNet 0.5x g=8| 2017 | Paper       | 57.700 |        |          |           |
|ShuffleNet 1.0x g=3| 2017 | Paper       | 67.400 |        |   4.3230 |  1924.415 |
|ShuffleNet 1.0x g=8| 2017 | Paper       | 67.600 |        |   4.8413 |  1912.766 |
|ShuffleNet V2 x0.5 | 2018 | torchvision | 60.552 | 81.746 |   1.3668 |    42.524 |
|ShuffleNet V2 x1.0 | 2018 | torchvision | 69.362 | 88.316 |   2.2786 |   148.808 |
|ShuffleNet V2 x1.5 | 2018 | Paper       | 72.600 |        |   3.5036 |   301.294 |
|ShuffleNet V2 x2.0 | 2018 | Paper       | 74.900 |        |   7.3940 |   590.741 |
|MNASNet 0.5        | 2018 | torchvision | 67.734 | 87.490 |  23.8346 |   110.587 |
|MNASNet 1.0        | 2018 | torchvision | 73.456 | 91.510 |   4.3833 |   325.329 |
|MobileNet V3 Small | 2019 | torchvision | 67.668 | 87.402 |   2.5429 |    59.368 |
|EfficientNet-B0    | 2019 | Paper       | 77.300 | 93.500 |   5.2885 |   401.679 |
|EfficientNet-B1    | 2019 | Paper       | 79.200 | 94.500 |   7.7942 |   591.948 |
|EfficientNet-B2    | 2019 | Paper       | 80.300 | 95.000 |   9.1100 |   682.357 |
|EfficientNet-B3    | 2019 | Paper       | 81.700 | 95.600 |  12.2332 |   993.680 |
|EfficientNet-B4    | 2019 | Paper       | 83.000 | 96.300 |  19.3416 |  1544.606 |
|EfficientNet-B5    | 2019 | Paper       | 83.700 | 96.700 |  30.3898 |  2413.021 |
|EfficientNet-B6    | 2019 | Paper       | 84.200 | 96.800 |  43.0407 |  3432.558 |
|EfficientNet-B7    | 2019 | Paper       | 84.400 |        |  66.3480 |  5267.130 |
|EfficientNetV2-S   | 2021 | Paper       | 83.900 |        |  22.0000 |           |
|EfficientNetV2-M   | 2021 | Paper       | 85.100 |        |  54.0000 |           |
|EfficientNetV2-L   | 2021 | Paper       | 85.700 |        | 120.0000 |           |
|EfficientNetV2-XL  | 2021 |             |        |        | 208.0000 |           |

> **MACs** : Multiply-Accumulate Operations
