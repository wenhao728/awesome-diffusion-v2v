# 📈 V2VBench
[(🚧 Comming soon) Leaderboard](./leaderboard.md)

## Installation
Clone the repository:
```bash
git clone https://github.com/wenhao728/awesome-diffusion-v2v.git
```

(Optional) We recommend using a virtual environment to manage the dependencies. You can refer [virtualenv](https://virtualenv.pypa.io/en/latest/user_guide.html) or [conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#activating-an-environment) to create a virtual environment.

Install the required packages:
```bash
pip install -r requirements.txt
```

Download the pre-trained models:
```bash
sh doc/install.sh
```
The `./checkpoints` directory should contain the following files:
```diff
 checkpoints
+ ├── DOVER
+ │   └── DOVER.pth
+ ├── gmflow
+ │   └── gmflow_sintel-0c07dcb3.pth
+ └── ViCLIP
+     ├── bpe_simple_vocab_16e6.txt.gz
+     └── ViClip-InternVid-10M-FLT.pth
```

## Run Evaluation

### (WIP) Single Video Evaluation
```python
```

### Batch Evaluation
Check the [Prepare Data](#optional-prepare-data) section to prepare the data for batch evaluation.
Then, simply run the following snippet to evaluate the edited videos:
```python
import logging

from evaluate import EvaluatorWrapper

logging.basicConfig(
    level=logging.INFO,  # Change it to logging.DEBUG if you want to troubleshoot
    format="%(asctime)s [%(levelname)s] %(message)s (%(filename)s:%(lineno)d)",
    handlers=[logging.StreamHandler(sys.stdout),]
)

evaluation = EvaluatorWrapper(
    index_file='data_example/config.yaml',
    edit_video_dir='data_example/edited_videos',
    reference_video_dir='/path/to/source_videos',
    metrics=[
        'aesthetic_score', 'clip_consistency', 'dino_consistency', 'dover_score',
        'clip_text_alignment', 'pick_score', 'viclip_text_alignment', 'motion_alignment',
    ], # or 'all' for all metrics
)
results = evaluation.evaluate(output_dir='/path/to/write/results')
```

## (Optional) Prepare Data
To evaluate a batch of editing results, the data should be organized in specified formats. 
You can download the V2VBench dataset from [(🚧 Comming soon) Download Dataset](https://path/to/v2vbench) or use your customized data with the following format.

### Configuration
The configuration file is a YAML file. And each video should be one entry of the `data` list. The following is an example of one entry:
```yaml
video_id: hike  # source video id
edit:
- prompt: a superman with a backpack hikes through the desert  # edit prompt 0
- prompt: a man with a backpack hikes through the dolomites, pixel art  # edit prompt 1
# ... more edit prompts
```
### Source Videos
The source videos for reference can be provided either as `video files` or as `directories containing video frame images` with filenames indicating the frame number. The video files or directories should be named according to the `video_id` specified in the configuration file.

### Edited Videos
For evaluation, the edited videos or video frame directories should be placed in the directory named according to the `video_id` specified in the configuration file. Each edited video should be named according to the edit prompt index.


## Shoutouts
This repository is inspired by the following open-source projects:
- [transformers](https://github.com/huggingface/transformers) by [Huggingface](https://github.com/huggingface)
- [LAION Aesthetic Predictor](https://github.com/LAION-AI/aesthetic-predictor) (and its [out-of-the-box implementation](https://github.com/shunk031/simple-aesthetics-predictor))
- [CLIP](https://github.com/openai/CLIP)
- [DINO-v2](https://github.com/facebookresearch/dinov2)
- [DOVER](https://github.com/VQAssessment/DOVER)
- [GMFlow](https://github.com/haofeixu/gmflow)
- [ViCLIP](https://github.com/OpenGVLab/InternVideo) 
- [PickScore](https://github.com/yuvalkirstain/PickScore)

We extend our gratitude to the authors for their exceptional work and open-source contributions.