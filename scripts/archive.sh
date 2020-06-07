torch-model-archiver --model-name resnet34 \
--version 1.0 \
--serialized-file resnet34.pt \
--extra-files ./index_to_name.json,./MyHandler.py \
--handler my_handler.py  \
--export-path model-store -f 