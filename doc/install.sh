# DOVER
# install git lfs to pull the checkpoints from huggingface
git lfs install
git clone https://huggingface.co/teowu/DOVER checkpoints/

# ViCLIP
# tokenizers
wget https://raw.githubusercontent.com/openai/CLIP/main/clip/bpe_simple_vocab_16e6.txt.gz \
    -P checkpoints/ViCLIP
# model weights
wget https://pjlab-gvm-data.oss-cn-shanghai.aliyuncs.com/internvideo/viclip/ViClip-InternVid-10M-FLT.pth \
    -P checkpoints/ViCLIP

# GMFlow
# download the pretrained model from google drive
gdown 1d5C5cgHIxWGsFR1vYs5XrQbbUiZl9TX2 -O checkpoints/
# unzip the model and move it to the correct directory
unzip -n checkpoints/pretrained.zip -d checkpoints/
mv checkpoints/pretrained checkpoints/gmflow
rm checkpoints/pretrained.zip