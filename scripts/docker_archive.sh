#!/bin/bash
docker run --rm \
    -v $(pwd):/  \
    pytorch/torchserve:0.1-cpu \
    torch-model-archiver --model-name resnet34 \
    --version 1.0 \
    --serialized-file resnet34.pt \
    --extra-files ./index_to_name.json \
    --handler image_classifier  \
    --export-path model-store -f 
