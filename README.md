<h1 align="center">Diffusion Model-Based Video Editing: A Survey</h1>

<p align="center">
<a href="https://github.com/wenhao728/awesome-diffusion-v2v"><img  src="https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg" ></a>
<a href="https://arxiv.org/abs/2407.07111"><img  src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a>
<a href="./doc/README.md"><img src="https://img.shields.io/badge/benchmark-Dataset-blue?style=flat"></a>
<a href="./doc/leaderboard.md"><img src="https://img.shields.io/badge/benchmark-Leaderboard-1abc9c?style=flat"></a>
<!-- <a href="https://opensource.org/license/mit/"><img  src="https://img.shields.io/badge/license-MIT-blue"></a> -->
<img  alt="GitHub last commit" src="https://img.shields.io/github/last-commit/wenhao728/awesome-diffusion-v2v?style=social"></a>
<!-- <img alt="GitHub watchers" src="https://img.shields.io/github/watchers/wenhao728/awesome-diffusion-v2v?style=social"> -->
<!-- <img alt="GitHub stars" src="https://img.shields.io/github/stars/wenhao728/awesome-diffusion-v2v?style=social"></a> -->
</p>

<p align="center">
<a href="https://github.com/wenhao728">Wenhao Sun</a>,
<a href=https://github.com/rongchengtu1>Rong-Cheng Tu</a>,
<a>Jingyi Liao</a>,
<a>Dacheng Tao</a>
<br>
<em>Nanyang Technological University</em>
</p>

<!-- <p align="center">
<img src="asset/teaser.gif" width="1024px"/>
</p> -->


https://github.com/wenhao728/awesome-diffusion-v2v/assets/65353366/fd42e40f-265d-4d72-8dc1-bf74d00fe87b

## 🍻 Citation
If you find this repository helpful, please consider citing our paper:
```bibtex
@article{sun2024v2vsurvey,
    author = {Wenhao Sun and Rong-Cheng Tu and Jingyi Liao and Dacheng Tao},
    title = {Diffusion Model-Based Video Editing: A Survey},
    journal = {CoRR},
    volume = {abs/2407.07111},
    year = {2024}
}
```


## 📌 Introduction
<p align="center">
<img src="asset/taxonomy-repo.png" width="85%">
<br><em>Overview of diffusion-based video editing model components.</em>
</p>

<!-- The diffusion process defines a Markov chain that progressively adds random noise to data and learns to reverse this process to generate desired data samples from noise. Deep neural networks facilitate the transitions between latent states. -->
- [Network and Training Paradigm](#network-and-training-paradigm) | network and data modifications
    - [Temporal Adaption](#temporal-adaption)
    - [Structure Conditioning](#structure-conditioning)
    - [Training Modification](#training-modification)
- [Attention Feature Injection](#attention-feature-injection) | a class of training-free techniques
    - [Inversion-Based Feature Injection](#inversion-based-feature-injection)
    - [Motion-Based Feature Injection](#motion-based-feature-injection)
- [Diffusion Latents Manipulation](#diffusion-latents-manipulation) | manipulate diffusion process
    - [Latent Initialization](#latent-initialization)
    - [Latent Transition](#latent-transition)
- [Canonical Representation](#canonical-representation) | the efficient video representation
- [Novel Conditioning](#novel-conditioning) | condtional video editing
    - [Point-Based Editing](#point-based-editing)
    - [Pose-Guided Human Action Editing](#pose-guided-human-action-edit)

> [!TIP]
> The papers are listed in reverse chronological order, formatted as follows:
> (Conference/Journal Year) Title, *Authors*

## Network and Training Paradigm
### Temporal Adaption

- (ECCV 24') Video Editing via Factorized Diffusion Distillation, *Singer et al.* 
    - <a href="https://arxiv.org/abs/2403.09334"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a> <a href="https://fdd-video-edit.github.io/"><img src="https://img.shields.io/badge/project-Website-%23009BD5"></a>
- (CVPR 24') **MaskINT**: Video Editing via Interpolative Non-autoregressive Masked Transformers, *Ma et al.* 
    - <a href="https://arxiv.org/abs/2312.12468"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a> <a href="https://maskint.github.io/"><img src="https://img.shields.io/badge/project-Website-%23009BD5"></a>
- (Preprint 23') **Fairy**: Fast Parallelized Instruction-Guided Video-to-Video Synthesis, *Wu et al.* 
    - <a href="https://arxiv.org/abs/2312.13834"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a> <a href="https://fairy-video2video.github.io/"><img src="https://img.shields.io/badge/project-Website-%23009BD5"></a>
- (CVPR 24') VidToMe: Video Token Merging for Zero-Shot Video Editing, *Li et al.* 
    - <a href="https://arxiv.org/abs/2312.10656"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a> <a href="https://vidtome-diffusion.github.io/"><img src="https://img.shields.io/badge/project-Website-%23009BD5"></a> <a href="https://github.com/lixirui142/VidToMe"><img src="https://img.shields.io/github/stars/lixirui142/VidToMe?style=social"></a>
- (CVPR 24') **SimDA**: Simple Diffusion Adapter for Efficient Video Generation, *Xing et al.* 
    - <a href="https://arxiv.org/abs/2308.09710"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a> <a href="https://chenhsing.github.io/SimDA/"><img src="https://img.shields.io/badge/project-Website-%23009BD5"></a> <a href="https://github.com/ChenHsing/SimDA"><img src="https://img.shields.io/github/stars/ChenHsing/SimDA?style=social"></a>
- (NeurIPS 23') Towards Consistent Video Editing with Text-to-Image Diffusion Models, *Zhang et al.* 
    - <a href="https://arxiv.org/abs/2305.17431"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a>
- (ICCV 23') **Tune-A-Video**: One-Shot Tuning of Image Diffusion Models for Text-to-Video Generation, *Wu et al.*
    - <a href="https://arxiv.org/abs/2212.11565"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a> <a href="https://tuneavideo.github.io/"><img src="https://img.shields.io/badge/project-Website-%23009BD5"></a> <a href="https://github.com/showlab/Tune-A-Video"><img src="https://img.shields.io/github/stars/showlab/Tune-A-Video?style=social"></a>


<p align="right">(<a href="#top">back to top</a>)</p>

### Structure Conditioning
- (Preprint '24) **EVA**: Zero-shot Accurate Attributes and Multi-Object Video Editing, *Yang et al.*
    - <a href="https://arxiv.org/abs/2403.16111"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a> <a href="https://knightyxp.github.io/EVA/"><img src="https://img.shields.io/badge/project-Website-%23009BD5"></a> <a href="https://github.com/knightyxp/EVA_Video_Edit"><img src="https://img.shields.io/github/stars/knightyxp/EVA_Video_Edit?style=social"></a>
- (IJCAI '24) **Diffutoon**: High-Resolution Editable Toon Shading via Diffusion Models, *Duan et al.*
    - <a href="https://arxiv.org/abs/2401.16224"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a> <a href="https://ecnu-cilab.github.io/DiffutoonProjectPage/"><img src="https://img.shields.io/badge/project-Website-%23009BD5"></a>
- (CVPR '24) **FlowVid**: Taming Imperfect Optical Flows for Consistent Video-to-Video Synthesis, *Liang et al.*
    - <a href="https://arxiv.org/abs/2312.17681"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a> <a href="https://jeff-liangf.github.io/projects/flowvid/"><img src="https://img.shields.io/badge/project-Website-%23009BD5"></a> <a href="https://github.com/Jeff-LiangF/FlowVid"><img src="https://img.shields.io/github/stars/Jeff-LiangF/FlowVid?style=social"></a>
- (Preprint '23) Motion-Conditioned Image Animation for Video Editing, *Yan et al.*
    - <a href="https://arxiv.org/abs/2311.18827"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a> <a href="https://facebookresearch.github.io/MoCA/"><img src="https://img.shields.io/badge/project-Website-%23009BD5"></a> <a href="https://github.com/facebookresearch/MoCA"><img src="https://img.shields.io/github/stars/facebookresearch/MoCA?style=social"></a>
- (CVPR '24) **LAMP**: Learn A Motion Pattern for Few-Shot-Based Video Generation, *Wu et al.*
    - <a href="https://arxiv.org/abs/2310.10769"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a> <a href="https://rq-wu.github.io/projects/LAMP/index.html"><img src="https://img.shields.io/badge/project-Website-%23009BD5"></a> <a href="https://github.com/RQ-Wu/LAMP"><img src="https://img.shields.io/github/stars/RQ-Wu/LAMP?style=social"></a>
- (ICLR '24) **Ground-A-Video**: Zero-shot Grounded Video Editing using Text-to-image Diffusion Models, *Jeong and Ye*
    - <a href="https://arxiv.org/abs/2310.01107"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a> <a href="https://ground-a-video.github.io/"><img src="https://img.shields.io/badge/project-Website-%23009BD5"></a> <a href="https://github.com/Ground-A-Video/Ground-A-Video"><img src="https://img.shields.io/github/stars/Ground-A-Video/Ground-A-Video?style=social"></a>
- (Preprint '23) **CCEdit**: Creative and Controllable Video Editing via Diffusion Models, *Feng et al.*
    - <a href="https://arxiv.org/abs/2309.16496"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a> <a href="https://ruoyufeng.github.io/CCEdit.github.io/"><img src="https://img.shields.io/badge/project-Website-%23009BD5"></a> <a href="https://github.com/RuoyuFeng/CCEdit"><img src="https://img.shields.io/github/stars/RuoyuFeng/CCEdit?style=social"></a>
- (Preprint '23) **MagicEdit**: High-Fidelity and Temporally Coherent Video Editing, *Liew et al.*
    - <a href="https://arxiv.org/abs/2308.14749"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a> <a href="https://magic-edit.github.io/"><img src="https://img.shields.io/badge/project-Website-%23009BD5"></a> <a href="https://github.com/magic-research/magic-edit"><img src="https://img.shields.io/github/stars/magic-research/magic-edit?style=social"></a>
- (Preprint '23) **VideoControlNet**: A Motion-Guided Video-to-Video Translation Framework by Using Diffusion Model with ControlNet, *Hu and Xu*
    - <a href="https://arxiv.org/abs/2307.14073"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a> <a href="https://github.com/ZhihaoHu/VideoControlNet"><img src="https://img.shields.io/github/stars/ZhihaoHu/VideoControlNet?style=social"></a>
- (NeurIPS '23) **VideoComposer**: Compositional Video Synthesis with Motion Controllability, *Wang et al.*
    - <a href="https://arxiv.org/abs/2306.02018"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a> <a href="https://videocomposer.github.io/"><img src="https://img.shields.io/badge/project-Website-%23009BD5"></a> <a href="https://github.com/ali-vilab/videocomposer"><img src="https://img.shields.io/github/stars/ali-vilab/videocomposer?style=social"></a>
- (Preprint '23) Structure and Content-Guided Video Synthesis with Diffusion Models, *Esser et al.*
    - <a href="https://arxiv.org/abs/2302.03011"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a> <a href="https://research.runwayml.com/gen1"><img src="https://img.shields.io/badge/project-Website-%23009BD5"></a>

<p align="right">(<a href="#top">back to top</a>)</p>

### Training Modification

- (Preprint 24') **Movie Gen**: A Cast of Media Foundation Models, *Polyak et al.*
    - <a href="https://arxiv.org/abs/2410.13720"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a> <a href="https://ai.meta.com/research/movie-gen/"><img src="https://img.shields.io/badge/project-Website-%23009BD5"></a> <a href="https://github.com/facebookresearch/MovieGenBench"><img src="https://img.shields.io/github/stars/facebookresearch/MovieGenBench?style=social"></a>
- (Preprint 24') **Still-Moving**: Customized Video Generation without Customized Video Data, *Chefer et al.*
    - <a href="https://arxiv.org/abs/2407.08674"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a> <a href="https://still-moving.github.io"><img src="https://img.shields.io/badge/project-Website-%23009BD5"></a> <a href="https://github.com/harshbhatt7585/StillMoving?tab=readme-ov-file"><img src="https://img.shields.io/github/stars/harshbhatt7585/StillMoving?style=social"></a>
- (Preprint 24') **EffiVED**: Efficient Video Editing via Text-instruction Diffusion Models, *Zhang et al.*
    - <a href="https://arxiv.org/abs/2403.11568"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a> <a href="https://github.com/alibaba/EffiVED"><img src="https://img.shields.io/github/stars/alibaba/EffiVED?style=social"></a>
- (ECCV 24') **Customize-A-Video**: One-Shot Motion Customization of Text-to-Video Diffusion Models, *Ren et al.*
    - <a href="https://arxiv.org/abs/2402.14780"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a> <a href="https://arxiv.org/abs/2402.14780"><img src="https://img.shields.io/badge/project-Website-%23009BD5"></a> <a href="https://github.com/customize-a-video/customize-a-video"><img src="https://img.shields.io/github/stars/customize-a-video/customize-a-video?style=social"></a>
- (Preprint 24') **VASE**: Object-Centric Appearance and Shape Manipulation of Real Videos, *Peruzzo et al.*
    - <a href="https://arxiv.org/abs/2401.02473"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a> <a href="https://helia95.github.io/vase-website/"><img src="https://img.shields.io/badge/project-Website-%23009BD5"></a> <a href="https://github.com/helia95/VASE"><img src="https://img.shields.io/github/stars/helia95/VASE?style=social"></a>
- (Preprint 23') **Customizing Motion in Text-to-Video Diffusion Models**, *Materzynska et al.*
    - <a href="https://arxiv.org/abs/2312.04966"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a> <a href="https://joaanna.github.io/customizing_motion/"><img src="https://img.shields.io/badge/project-Website-%23009BD5"></a>
- (ECCV 24') **SAVE**: Protagonist Diversification with Structure Agnostic Video Editing, *Song et al.*
    - <a href="https://arxiv.org/abs/2312.02503"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a> <a href="https://ldynx.github.io/SAVE/"><img src="https://img.shields.io/badge/project-Website-%23009BD5"></a> <a href="https://github.com/ldynx/SAVE"><img src="https://img.shields.io/github/stars/ldynx/SAVE?style=social"></a>
- (CVPR 24') **VMC**: Video Motion Customization using Temporal Attention Adaption for Text-to-Video Diffusion Models, *Jeong et al.*
    - <a href="https://arxiv.org/abs/2312.00845"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a> <a href="https://video-motion-customization.github.io/"><img src="https://img.shields.io/badge/project-Website-%23009BD5"></a> <a href="https://github.com/HyeonHo99/Video-Motion-Customization"><img src="https://img.shields.io/github/stars/HyeonHo99/Video-Motion-Customization?style=social"></a>
- (ICLR 24') Consistent Video-to-Video Transfer Using Synthetic Dataset, *Cheng et al.*
    - <a href="https://arxiv.org/abs/2311.00213"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a> <a href="https://github.com/amazon-science/instruct-video-to-video/tree/main"><img src="https://img.shields.io/github/stars/amazon-science/instruct-video-to-video?style=social"></a>
- (Preprint 23') **VIDiff**: Translating Videos via Multi-Modal Instructions with Diffusion Models, *Xing et al.*
    - <a href="https://arxiv.org/abs/2311.18837"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a> <a href="https://chenhsing.github.io/VIDiff/"><img src="https://img.shields.io/badge/project-Website-%23009BD5"></a> <a href="https://github.com/ChenHsing/VIDiff?tab=readme-ov-file"><img src="https://img.shields.io/github/stars/ChenHsing/VIDiff?style=social"></a>
- (ECCV 24') **MotionDirector**: Motion Customization of Text-to-Video Diffusion Models, *Zhao et al.*
    - <a href="https://arxiv.org/abs/2310.08465"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a> <a href="https://showlab.github.io/MotionDirector/"><img src="https://img.shields.io/badge/project-Website-%23009BD5"></a> <a href="https://github.com/showlab/MotionDirector"><img src="https://img.shields.io/github/stars/showlab/MotionDirector?style=social"></a>
- (ICME 24') **InstructVid2Vid**: Controllable Video Editing with Natural Language Instructions, *Qin et al.*
    - <a href="https://arxiv.org/abs/2305.12328"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a>
- (Preprint 23') **Dreamix**: Video Diffusion Models are General Video Editors, *Molad et al.*
    - <a href="https://arxiv.org/abs/2302.01329"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a> <a href="https://dreamix-video-editing.github.io/"><img src="https://img.shields.io/badge/project-Website-%23009BD5"></a>

<p align="right">(<a href="#top">back to top</a>)</p>


## Attention Feature Injection

### Inversion-Based Feature Injection

- (TMLR 24') **AnyV2V**: A Tuning-Free Framework For Any Video-to-Video Editing Tasks, *Ku et al.*  
    - <a href="https://arxiv.org/abs/2403.14468"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a> <a href="https://tiger-ai-lab.github.io/AnyV2V/"><img src="https://img.shields.io/badge/project-Website-%23009BD5"></a> <a href="https://github.com/TIGER-AI-Lab/AnyV2V"><img src="https://img.shields.io/github/stars/TIGER-AI-Lab/AnyV2V?style=social"></a>
- (ECCV 24') **Object-Centric Diffusion**: Efficient Video Editing, *Kahatapitiya et al.*  
    - <a href="https://arxiv.org/abs/2401.05735"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a> <a href="https://qualcomm-ai-research.github.io/object-centric-diffusion/"><img src="https://img.shields.io/badge/project-Website-%23009BD5"></a>
- (Preprint 24') **UniEdit**: A Unified Tuning-Free Framework for Video Motion and Appearance Editing, *Bai et al.*  
    - <a href="https://arxiv.org/abs/2402.13185"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a> <a href="https://jianhongbai.github.io/UniEdit/"><img src="https://img.shields.io/badge/project-Website-%23009BD5"></a> <a href="https://github.com/JianhongBai/UniEdit"><img src="https://img.shields.io/github/stars/JianhongBai/UniEdit?style=social"></a>
- (Preprint 23') **Make-A-Protagonist**: Generic Video Editing with An Ensemble of Experts, *Zhao et al.*  
    - <a href="https://arxiv.org/abs/2305.08850"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a> <a href="https://make-a-protagonist.github.io/"><img src="https://img.shields.io/badge/project-Website-%23009BD5"></a> <a href="https://github.com/HeliosZhao/Make-A-Protagonist"><img src="https://img.shields.io/github/stars/HeliosZhao/Make-A-Protagonist?style=social"></a>
- (Preprint 23') Zero-Shot Video Editing Using Off-The-Shelf Image Diffusion Models, *Wang et al.*  
    - <a href="https://arxiv.org/abs/2303.17599"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a> <a href="https://github.com/baaivision/vid2vid-zero"><img src="https://img.shields.io/github/stars/baaivision/vid2vid-zero?style=social"></a>
- (ICCV 23') **FateZero**: Fusing Attentions for Zero-shot Text-based Video Editing, *Qi et al.*  
    - <a href="https://arxiv.org/abs/2303.09535"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a> <a href="https://fate-zero-edit.github.io/"><img src="https://img.shields.io/badge/project-Website-%23009BD5"></a> <a href="https://github.com/ChenyangQiQi/FateZero"><img src="https://img.shields.io/github/stars/ChenyangQiQi/FateZero?style=social"></a>
- (ACML 23') **Edit-A-Video**: Single Video Editing with Object-Aware Consistency, *Shin et al.*  
    - <a href="https://arxiv.org/abs/2303.07945"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a> <a href="https://edit-a-video.github.io/"><img src="https://img.shields.io/badge/project-Website-%23009BD5"></a>
- (Preprint 23') **Video-P2P**: Video Editing with Cross-attention Control, *Liu et al.*  
    - <a href="https://arxiv.org/abs/2303.04761"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a> <a href="https://video-p2p.github.io/"><img src="https://img.shields.io/badge/project-Website-%23009BD5"></a> <a href="https://github.com/dvlab-research/Video-P2P"><img src="https://img.shields.io/github/stars/dvlab-research/Video-P2P?style=social"></a>

<p align="right">(<a href="#top">back to top</a>)</p>

### Motion-Based Feature Injection
- (CVPR 24') **FRESCO**: Spatial-Temporal Correspondence for Zero-Shot Video Translation, *Yang et al.*  
    - <a href="https://arxiv.org/abs/2403.12962"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a> <a href="https://www.mmlab-ntu.com/project/fresco/"><img src="https://img.shields.io/badge/project-Website-%23009BD5"></a> <a href="https://github.com/williamyang1991/FRESCO"><img src="https://img.shields.io/github/stars/williamyang1991/FRESCO?style=social"></a>
- (ICLR 24') **FLATTEN**: optical FLow-guided ATTENtion for consistent text-to-video editing, *Cong et al.*  
    - <a href="https://arxiv.org/abs/2310.05922"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a> <a href="https://flatten-video-editing.github.io/"><img src="https://img.shields.io/badge/project-Website-%23009BD5"></a> <a href="https://github.com/yrcong/flatten"><img src="https://img.shields.io/github/stars/yrcong/flatten?style=social"></a>
- (ICLR 24') **TokenFlow**: Consistent Diffusion Features for Consistent Video Editing, *Geyer et al.*  
    - <a href="https://arxiv.org/abs/2307.10373"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a> <a href="https://diffusion-tokenflow.github.io/"><img src="https://img.shields.io/badge/project-Website-%23009BD5"></a> <a href="https://github.com/omerbt/TokenFlow"><img src="https://img.shields.io/github/stars/omerbt/TokenFlow?style=social"></a>


<p align="right">(<a href="#top">back to top</a>)</p>

## Diffusion Latents Manipulation

### Latent Initialization
- (CVPR 24') **A Video is Worth 256 Bases**: Spatial-Temporal Expectation-Maximization Inversion for Zero-Shot Video Editing, *Li et al.*  
    - <a href="https://arxiv.org/abs/2312.05856"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a> <a href="https://stem-inv.github.io/page/"><img src="https://img.shields.io/badge/project-Website-%23009BD5"></a> <a href="https://github.com/STEM-Inv/stem-inv"><img src="https://img.shields.io/github/stars/STEM-Inv/stem-inv?style=social"></a>
- (Preprint 23') **Video ControlNet**: Towards Temporally Consistent Synthetic-to-Real Video Translation Using Conditional Image Diffusion Models, *Chu et al.*  
    - <a href="https://arxiv.org/abs/2305.19193"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a>
- (Preprint 23') **Control-A-Video**: Controllable Text-to-Video Generation with Diffusion Models, *Chen et al.*  
    - <a href="https://arxiv.org/abs/2305.13840"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a> <a href="https://controlavideo.github.io/"><img src="https://img.shields.io/badge/project-Website-%23009BD5"></a> <a href="https://github.com/Weifeng-Chen/control-a-video"><img src="https://img.shields.io/github/stars/Weifeng-Chen/control-a-video?style=social"></a>
- (ICCV 23') **Text2Video-Zero**: Text-to-Image Diffusion Models are Zero-Shot Video Generators, *Khachatryan et al.*  
    - <a href="https://arxiv.org/abs/2303.13439"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a> <a href="https://text2video-zero.github.io/"><img src="https://img.shields.io/badge/project-Website-%23009BD5"></a> <a href="https://github.com/Picsart-AI-Research/Text2Video-Zero"><img src="https://img.shields.io/github/stars/Picsart-AI-Research/Text2Video-Zero?style=social"></a>


<p align="right">(<a href="#top">back to top</a>)</p>

### Latent Transition
- (CVPR 24') **GenVideo**: One-shot target-image and shape-aware video editing using T2I diffusion models, *Harsha et al.*
    - <a href="https://arxiv.org/abs/2404.12541"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a>
- (Preprint 24') **MotionClone**: Training-Free Motion Cloning for Controllable Video Generation, *Ling et al.*
    - <a href="https://arxiv.org/abs/2406.05338"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a> <a href="https://bujiazi.github.io/motionclone.github.io/"><img src="https://img.shields.io/badge/project-Website-%23009BD5"></a> <a href="https://github.com/Bujiazi/MotionClone/"><img src="https://img.shields.io/github/stars/Bujiazi/MotionClone?style=social"></a>
- (CVPR 24') **RAVE**: Randomized Noise Shuffling for Fast and Consistent Video Editing with Diffusion Models, *Kara et al.*
    - <a href="https://arxiv.org/abs/2312.04524"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a> <a href="https://rave-video.github.io/"><img src="https://img.shields.io/badge/project-Website-%23009BD5"></a> <a href="https://github.com/RehgLab/RAVE"><img src="https://img.shields.io/github/stars/RehgLab/RAVE?style=social"></a>
- (CVPR 24') Space-Time Diffusion Features for Zero-Shot Text-Driven Motion Transfer, *Yatim et al.*
    - <a href="https://arxiv.org/abs/2311.17009"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a> <a href="https://diffusion-motion-transfer.github.io/"><img src="https://img.shields.io/badge/project-Website-%23009BD5"></a> <a href="https://github.com/diffusion-motion-transfer/diffusion-motion-transfer"><img src="https://img.shields.io/github/stars/diffusion-motion-transfer/diffusion-motion-transfer?style=social"></a>
- (Preprint 23') **DiffSynth**: Latent In-Iteration Deflickering for Realistic Video Synthesis, *Duan et al.*
    - <a href="https://arxiv.org/abs/2308.03463"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a> <a href="https://anonymous456852.github.io/"><img src="https://img.shields.io/badge/project-Website-%23009BD5"></a> <a href="https://github.com/alibaba/EasyNLP/tree/master/diffusion/DiffSynth"><img src="https://img.shields.io/github/stars/alibaba/EasyNLP?style=social"></a>
- (SIGGRAPH 23') **Rerender A Video**: Zero-Shot Text-Guided Video-to-Video Translation, *Yang et al.*
    - <a href="https://arxiv.org/abs/2306.07954"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a> <a href="https://www.mmlab-ntu.com/project/rerender/"><img src="https://img.shields.io/badge/project-Website-%23009BD5"></a> <a href="https://github.com/williamyang1991/Rerender_A_Video"><img src="https://img.shields.io/github/stars/williamyang1991/Rerender_A_Video?style=social"></a>
- (ICLR 23') **ControlVideo**: Training-free Controllable Text-to-Video Generation, *Zhang et al.*
    - <a href="https://arxiv.org/abs/2305.13077"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a> <a href="https://controlvideov1.github.io/"><img src="https://img.shields.io/badge/project-Website-%23009BD5"></a> <a href="https://github.com/YBYBZhang/ControlVideo"><img src="https://img.shields.io/github/stars/YBYBZhang/ControlVideo?style=social"></a>
- (ICCV 23') **Pix2Video**: Video Editing using Image Diffusion, *Ceylan et al.*
    - <a href="https://arxiv.org/abs/2303.12688"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a> <a href="https://duyguceylan.github.io/pix2video.github.io/"><img src="https://img.shields.io/badge/project-Website-%23009BD5"></a> <a href="https://github.com/duyguceylan/pix2video"><img src="https://img.shields.io/github/stars/duyguceylan/pix2video?style=social"></a>

<p align="right">(<a href="#top">back to top</a>)</p>

## Canonical Representation
- (Preprint 23') **Neural Video Fields Editing**: Neural Video Fields Editing, *Yang et al.*  
    - <a href="https://arxiv.org/abs/2312.08882"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a> <a href="https://nvedit.github.io/"><img src="https://img.shields.io/badge/project-Website-%23009BD5"></a> <a href="https://github.com/Ysz2022/NVEdit"><img src="https://img.shields.io/github/stars/Ysz2022/NVEdit?style=social"></a>
- (Preprint 23') **DiffusionAtlas**: High-Fidelity Consistent Diffusion Video Editing, *Chang et al.*  
    - <a href="https://arxiv.org/abs/2312.03772"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a> <a href="https://diffusionatlas.github.io/"><img src="https://img.shields.io/badge/project-Website-%23009BD5"></a> 
- (ICCV 23') **StableVideo**: Text-driven Consistency-aware Diffusion Video Editing, *Chai et al.*  
    - <a href="https://arxiv.org/abs/2308.09592"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a> <a href="https://github.com/rese1f/StableVideo"><img src="https://img.shields.io/github/stars/rese1f/StableVideo?style=social"></a>  
- (CVPR 24') **CoDeF**: Content Deformation Fields for Temporally Consistent Video Processing, *Ouyang et al.*  
    - <a href="https://arxiv.org/abs/2308.07926"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a> <a href="https://qiuyu96.github.io/CoDeF/"><img src="https://img.shields.io/badge/project-Website-%23009BD5"></a> <a href="https://github.com/qiuyu96/CoDeF"><img src="https://img.shields.io/github/stars/qiuyu96/CoDeF?style=social"></a>
- (TMLR 24') **VidEdit**: Zero-Shot and Spatially Aware Text-Driven Video Editing, *Couairon et al.*  
    - <a href="https://arxiv.org/abs/2306.08707"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a> <a href="https://videdit.github.io/"><img src="https://img.shields.io/badge/project-Website-%23009BD5"></a>  
- (CVPR 23') Shape-aware Text-driven Layered Video Editing, *Lee et al.*  
    - <a href="https://openaccess.thecvf.com/content/CVPR2023/papers/Lee_Shape-Aware_Text-Driven_Layered_Video_Editing_CVPR_2023_paper.pdf"><img src="https://img.shields.io/badge/Open%20Access-Paper-%23FFA500"></a> <a href="https://text-video-edit.github.io/"><img src="https://img.shields.io/badge/project-Website-%23009BD5"></a> <a href="https://github.com/text-video-edit/shape-aware-text-driven-layered-video-editing-release"><img src="https://img.shields.io/github/stars/text-video-edit/shape-aware-text-driven-layered-video-editing-release?style=social"></a>  



<p align="right">(<a href="#top">back to top</a>)</p>

## Novel Conditioning

### Point-Based Editing

- (SIGGRAPH 24') **MotionCtrl**: A Unified and Flexible Motion Controller for Video Generation, *Wang et al.*
    - <a href="https://arxiv.org/abs/2312.03641"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a> <a href="https://wzhouxiff.github.io/projects/MotionCtrl/"><img src="https://img.shields.io/badge/project-Website-%23009BD5"></a> <a href="https://github.com/TencentARC/MotionCtrl"><img src="https://img.shields.io/github/stars/TencentARC/MotionCtrl?style=social"></a>
- (Preprint 23') **Drag-A-Video**: Non-rigid Video Editing with Point-based Interaction, *Teng et al.*
    - <a href="https://arxiv.org/abs/2312.02936"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a>
- (ECCV 24') **DragVideo**: Interactive Drag-style Video Editing, *Deng et al.*
    - <a href="https://arxiv.org/abs/2312.02216"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a> <a href="https://github.com/RickySkywalker/DragVideo-Official"><img src="https://img.shields.io/github/stars/RickySkywalker/DragVideo-Official?style=social"></a>
- (CVPR 24') **VideoSwap**: Customized Video Subject Swapping with Interactive Semantic Point Correspondence, *Placeholder et al.*
    - <a href="https://arxiv.org/abs/2312.02087"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a> <a href="https://videoswap.github.io/"><img src="https://img.shields.io/badge/project-Website-%23009BD5"></a> <a href="https://github.com/showlab/VideoSwap"><img src="https://img.shields.io/github/stars/showlab/VideoSwap?style=social"></a>


<p align="right">(<a href="#top">back to top</a>)</p>

### Pose-Guided Human Action Editing
- (SCIS 24') **UniAnimate**: Taming Unified Video Diffusion Models for Consistent Human Image Animation, *Wang et al.*  
    - <a href="https://arxiv.org/abs/2406.01188"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a> <a href="https://unianimate.github.io/"><img src="https://img.shields.io/badge/project-Website-%23009BD5"></a> <a href="https://github.com/ali-vilab/UniAnimate"><img src="https://img.shields.io/github/stars/ali-vilab/UniAnimate?style=social"></a>
- (Preprint 24') **Zero-shot High-fidelity and Pose-controllable Character Animation**, *Zhu et al.*  
    - <a href="https://arxiv.org/abs/2404.13680"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a>
- (Preprint 23') **Animate Anyone**: Consistent and Controllable Image-to-Video Synthesis for Character Animation, *Hu et al.*  
    - <a href="https://arxiv.org/abs/2311.17117"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a> <a href="https://humanaigc.github.io/animate-anyone/"><img src="https://img.shields.io/badge/project-Website-%23009BD5"></a> <a href="https://github.com/HumanAIGC/AnimateAnyone"><img src="https://img.shields.io/github/stars/HumanAIGC/AnimateAnyone?style=social"></a> <a href="https://github.com/MooreThreads/Moore-AnimateAnyone"><img src="https://img.shields.io/github/stars/MooreThreads/Moore-AnimateAnyone?style=social"></a>
- (Preprint 23') **MagicAnimate**: Temporally Consistent Human Image Animation using Diffusion Model, *Xu et al.*  
    - <a href="https://arxiv.org/abs/2311.16498"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a> <a href="https://showlab.github.io/magicanimate/"><img src="https://img.shields.io/badge/project-Website-%23009BD5"></a> <a href="https://github.com/magic-research/magic-animate"><img src="https://img.shields.io/github/stars/magic-research/magic-animate?style=social"></a>
- (ICML 24') **MagicPose**: Realistic Human Poses and Facial Expressions Retargeting with Identity-aware Diffusion, *Chang et al.*  
    - <a href="https://arxiv.org/abs/2311.12052"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a> <a href="https://boese0601.github.io/magicdance/"><img src="https://img.shields.io/badge/project-Website-%23009BD5"></a> <a href="https://github.com/Boese0601/MagicDance"><img src="https://img.shields.io/github/stars/Boese0601/MagicDance?style=social"></a>
- (CVPR 24') **DisCo**: Disentangled Control for Realistic Human Dance Generation, *Wang et al.*  
    - <a href="https://arxiv.org/abs/2307.00040"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a> <a href="https://disco-dance.github.io/"><img src="https://img.shields.io/badge/project-Website-%23009BD5"></a> <a href="https://github.com/Wangt-CN/DisCo"><img src="https://img.shields.io/github/stars/Wangt-CN/DisCo?style=social"></a>
- (ICCV 23') **DreamPose**: Fashion Image-to-Video Synthesis via Stable Diffusion, *Karras et al.*  
    - <a href="https://arxiv.org/abs/2304.06025"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a> <a href="https://grail.cs.washington.edu/projects/dreampose/"><img src="https://img.shields.io/badge/project-Website-%23009BD5"></a> <a href="https://github.com/johannakarras/DreamPose"><img src="https://img.shields.io/github/stars/johannakarras/DreamPose?style=social"></a>
- (AAAI 23') **Follow Your Pose**: Pose-Guided Text-to-Video Generation using Pose-Free Videos, *Ma et al.*  
    - <a href="https://arxiv.org/abs/2304.01186"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a> <a href="https://follow-your-pose.github.io/"><img src="https://img.shields.io/badge/project-Website-%23009BD5"></a> <a href="https://github.com/mayuelala/FollowYourPose"><img src="https://img.shields.io/github/stars/mayuelala/FollowYourPose?style=social"></a>

<p align="right">(<a href="#top">back to top</a>)</p>


## 📜 Change Log
- [*28 Nov 2024*] Update the list format to enhance clarity.