# 📈 V2VBench
[Leaderboard](./leaderboard.md)

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
### Video format
Videos can be provided either as "video files" (e.g. `video_name.mp4`, `video_name.gif`) or "directories containing video frame images" with filenames indicating the frame number (e.g. `video_name/01.jpg`, `video_name/02.jpg`, ...).

### Single Video Evaluation
Run the following snippet to evaluate one edited video:
```python
import logging

from v2vbench import EvaluatorWrapper

logging.basicConfig(
    level=logging.INFO,  # Change it to logging.DEBUG if you want to troubleshoot
    format="%(asctime)s [%(levelname)s] %(message)s (%(filename)s:%(lineno)d)",
    handlers=[logging.StreamHandler(sys.stdout),]
)

evaluation = EvaluatorWrapper(metrics='all')  # 'all' for all metrics
# print(EvaluatorWrapper.all_metrics)  # list all available metrics

results = evaluation.evaluate(
    edit_video='/path/to/edited_video',
    reference_video='/path/to/source_video',
    edit_prompt='<edit prompt>',
    output_dir='/path/to/save/results',
)
```

### Batch Evaluation
Check the [Prepare Data](#optional-prepare-data) section to prepare the data for batch evaluation.
Then, simply run the following snippet to evaluate a batch of edited videos:
```python
import logging

from v2vbench import EvaluatorWrapper

logging.basicConfig(
    level=logging.INFO,  # Change it to logging.DEBUG if you want to troubleshoot
    format="%(asctime)s [%(levelname)s] %(message)s (%(filename)s:%(lineno)d)",
    handlers=[logging.StreamHandler(sys.stdout),]
)

evaluation = EvaluatorWrapper(metrics='all')  # 'all' for all metrics
# print(EvaluatorWrapper.all_metrics)  # list all available metrics

results = evaluation.evaluate(
    edit_video='/path/to/edited_videos',
    reference_video='/path/to/source_videos',
    index_file='/path/to/config.yaml',
    output_dir='/path/to/save/results',
    # it is recommended to cache the flow of source videos for motion_alignment to avoid redundant computation
    evaluator_kwargs={'motion_alignment': {'cache_flow': True}},
)
```

## (Optional) Prepare Data
To evaluate a batch of editing results, the data should be organized in specified formats. 
You can download the V2VBench dataset from [Huggingface Datasets](https://huggingface.co/datasets/Wenhao-Sun/V2VBench) or use your customized data with the following format.

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
The source videos for reference should be named according to the `video_id` specified in the configuration file.

### Edited Videos
For evaluation, the edited videos should be placed in the directory named according to the `video_id` specified in the configuration file. Each edited video should be named according to the edit prompt index.


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
- [VBench](https://github.com/Vchitect/VBench)

We extend our gratitude to the authors for their exceptional work and open-source contributions.