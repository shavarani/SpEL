import torch

from model import SpELAnnotator

device = torch.device("cpu")

annotator = SpELAnnotator()
annotator.load_checkpoint(None,
                          device=device,
                          load_from_torch_hub=True,
                          finetuned_after_step=4)