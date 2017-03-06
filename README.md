# cifar10neptune
Example of solving CIFAR-10 with Neptune.

It is not simples Neptune showcase, as this CIFAR-10 solution is pretty sophisticated. It uses train and test time data augmentation, batch normalization, drop-out. After training it scores above 90% of accuracy on test dataset.

Command to run the experiment with Neptune on your local computer:
```
neptune run cifar10neptune/main.py --paths-to-dump cifar10neptune --dump-dir-url ~/neptune_jobs --config cifar10neptune/cfg.yaml --shallow yes --use_batch_norm no
```

In order to have access to Neptune you can:

* use Neptune Go: http://go.neptune.deepsense.io
* or apply for the Early Adopters Program: https://deepsense.io/neptune-early-adopter-program/
