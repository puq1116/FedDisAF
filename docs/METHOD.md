# FedDisAF and combination boundaries

The client model computes `h = backbone(image)`, `x_p = generator(h)`, and
`x_s = h - x_p`. A shared classifier predicts local labels from `x_p`.
The generator is Linear–BatchNorm–ReLU–Linear. ResNet-18 uses 512-dimensional
features; the default initialization is ImageNet-pretrained.

During local training, independent Gaussian noise with standard deviation
`dp_noise_scale` is added to each visited shared vector. Detached noisy vectors
are collected across batches and local epochs, then averaged by class.
This is feature perturbation.

`FedDisAFPlugin.update_library()` groups uploads by class, computes their centroid,
and applies softmax to each prototype's cosine similarity to that centroid,
divided by `feddisaf_proto_temperature`. It stores a detached weighted sum on
the CPU. A singleton upload is returned directly; a class without an upload
retains its previous entry. The library starts with unavailable entries.

`FedDisAFPlugin.prototype_loss()` passes each available prototype through the
same classifier and averages cross-entropy across the available classes. The
classifier temporarily uses evaluation mode for singleton BatchNorm/dropout;
its parameters still receive gradients. The detached inputs prevent a direct
gradient through prototypes into the backbone or generator. The local objective
includes `proto_loss_weight * prototype_loss` in addition to the source
combination's local objective. Inference uses the private branch only.

## Responsibilities of the combinations

| Combination | Retained source-specific behavior |
|---|---|
| FedAvg + FedDisAF | Sample-count model averaging and local SGD |
| FedNova + FedDisAF | Source epoch-based coefficient and coefficient-weighted model averaging |
| FedProx + FedDisAF | Local proximal penalty; source uniform model averaging |
| SCAFFOLD + FedDisAF | Source combination path, including its unresolved control-variable issue |
| FedALA + FedDisAF | Source history/adaptation weighting and optional model-weight attention |
| FedFTG + FedDisAF | Source global-feature guidance, clipping, and model-weight attention |
| FedPop + FedDisAF | Source population/evolution weighting and model-weight attention |



