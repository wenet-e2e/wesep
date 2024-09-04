# Libri2Mix Recipe


## Goal of this recipe
This recipe aims to illustrate how to use WeSep to perform the target speaker extraction task on a pre-defined training set such as Libri2Mix, the mixtures have been prepared on the disk. If you want to check the online data processing and scale to larger training set, please check the voxceleb1 recipe.

## Difference of V1 and V2
The difference between v1 and v2 lies in the approach to speaker modeling.

- v1 outlines a process where WeSpeaker is used to extract embeddings beforehand, which are then saved to disk and sampled during training. This setup allows you to use other toolkits or existing speaker embeddings freely.
- v2 demonstrates a more integrated approach with the WeSpeaker toolkit (recommended). You can choose to either fix the speaker encoder or train it jointly with the separation backbone.

