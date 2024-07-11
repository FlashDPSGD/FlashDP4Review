## Flash-DP: Optimizing Large Language Models for Differential Privacy

Require:
```shell
PyTorch >= 2.2
CUDA >= 11.8
```

To install flashdp, just run
```shell
bash install.sh
```

Usage:
```python
from flashdp import wrap_model
# define your pytorch model on GPU
model = torch_model.cuda()
# wrap the pytorch model with our FlashDP
model = wrap_model(model, target_modules=[nn.Linear])
```
