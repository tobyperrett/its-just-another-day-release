# It's Just Another Day: Unique Video Captioning by Discriminative Prompting

This repo contains the code for the paper "It's Just Another Day: Unique Video Captioning by Discriminative Prompting".

[Project Website](https://tobyperrett.github.io/its-just-another-day/)

[arXiv](https://arxiv.org/abs/2209.13994)

This repo is forked from [LaViLa](https://github.com/facebookresearch/LaViLa/). Please follow their installation instructions.

## Quick Start on Egocentric Benchmark

We recommend downloading all the models/checkpoints/pre-extracted features to a directory called `storage` directory for simplicity.

### Download Ego4D Videos
First, you'll need to download videos from Ego4D. Once this is done, to preprocess them (i.e. resize to a sensible resolution and chop into 300s chunks for quick reading), use the following command and set the options at the top of the script.

```python
python preprocess_ego4d.py
```

This can be run on lots of CPU nodes in parallel with the following slurm script. Remember to change the options at the top of the script. 

```bash
sbatch lavila/slurm/bc4/submit_convert_ego4d.sh
```

For all subsequent scripts, you'll need to set the `--root` option to the directory where the preprocessed videos (or features) are stored.

### Download LaViLa checkpoints

Download the LaViLa video/text embedding network and VCLM checkpoints. The files you need are:

[vclm_openai_timesformer_large_336px_gpt2_xl.pt_ego4d.jobid_246897.ep_0003.md5sum_443263.pth](https://dl.fbaipublicfiles.com/lavila/checkpoints/narrator/vclm_openai_timesformer_large_336px_gpt2_xl.pt_ego4d.jobid_246897.ep_0003.md5sum_443263.pth)

[clip_openai_timesformer_large.narrator_rephraser.ep_0003.md5sum_c89337.pth](https://dl.fbaipublicfiles.com/lavila/checkpoints/dual_encoders/ego4d/clip_openai_timesformer_large.narrator_rephraser.ep_0003.md5sum_c89337.pth)

### Download EgoVLP checkpoints (as the evaluation space)

The uniqueness is evaluated in the EgoVLP video/text embedding space. Download the EgoVLP checkpoint from here:

[clip_openai_timesformer_large_336px_distilbert_base.baseline.ep_0003.pth](https://dl.fbaipublicfiles.com/lavila/checkpoints/dual_encoders/ego4d/clip_openai_timesformer_large_336px_distilbert_base.baseline.ep_0003.pth)

### Download CDPNet checkpoints

CDPNet predicts the similarity between video clips and prompted captions. Download the CDPNet checkpoints from here:

[arch10_CSPairClsP_p10_e30_model.pkl](https://drive.google.com/file/d/1-NCVWsZmLkxE9kB3Lo7bwtgk_LC7cG2c/view?usp=sharing)

### Benchmark

The clips used for the benchmark are contained in jsons. The first has the Ego4D video ID, timestamp, and original narration. The second groups them into sets. We use the first 300 sets for our benchmark, but we provide all 880 just in case they are useful. Because of the way LaViLa loads files, you'll need these three files in the same directory. Note that no training happens - train.pkl and val.pkl are just duplicates of the validation data:

[train_preds.pkl](https://drive.google.com/file/d/109_7mTwuB-MAP_AYl4-iGRUzxBQWiiCU/view?usp=sharing), [val_preds.pkl](https://drive.google.com/file/d/100IEts_Be4qAGFgCL4b60hMuNX9YxKr4/view?usp=sharing), [amin_clip_P1_OnlinePerVid_p10_e0.pkl](https://drive.google.com/file/d/10L7UG8dFX7m3ApHnsrHu5scXikTx6cqS/view?usp=sharing)

### Running the benchmarks

To run inference on a the egocentric benchmark for the T=+10 (i.e. the original 5s clip and advancing two) LaViLa + CDPNet, run the following command. Remember to use the full paths for the saved models.

```python
python main_run_unique_captioner.py --cap_resume storage/vclm_openai_timesformer_large_336px_gpt2_xl.pt_ego4d.jobid_246897.ep_0003.md5sum_443263.pth --pp_resume storage/arch10_CSPairClsP_p10_e30_model.pkl --emb_resume storage/clip_openai_timesformer_large.narrator_rephraser.ep_0003.md5sum_c89337.pth --eval_emb_resume storage/clip_openai_timesformer_large_336px_distilbert_base.baseline.ep_0003.pth --max_offset 10 --output_dir results --wbr cdp_lavila-vclm_egovlp-eval_10s
```

To run the LaViLa baseline for T=+10s, run the following. Notice how we specify `--fixed_clip_len 15` as this includes the original 5s clip and the 10s advancement.

```python
python main_run_unique_captioner.py --cap_resume storage/vclm_openai_timesformer_large_336px_gpt2_xl.pt_ego4d.jobid_246897.ep_0003.md5sum_443263.pth --pp_resume storage/arch10_CSPairClsP_p10_e30_model.pkl --emb_resume storage/clip_openai_timesformer_large.narrator_rephraser.ep_0003.md5sum_c89337.pth --eval_emb_resume storage/clip_openai_timesformer_large_336px_distilbert_base.baseline.ep_0003.pth --comb_maxp 1 --no_cap_default lav --fixed_clip_len 15 --output_dir results --wbr base_lavila-vclm_egovlp-eval_10s
```

This runs everything online with no pre-extracted features.

## Training CDPNet

If you want to train CDPNet instead of using our pre-trained one, you'll first need the embeddings of prompted captions from the training set with the LaViLa XL VCLM and Large embedding space as above. This CDPNet will be specific to these two LaViLa models and the set of prompts, so probably won't be that useful to you. However, if you do want them, you can either extract and embed all these yourself, or contact me and I'll find a way to get them to you (they're quite large). 

Next, run the following command:

```python
python main_train_prompt_predictor.py --output-str arch10
```

This reads in prompted captions generated by the VCLM, and trains CDPNet to predict the cosine similarity between video clips and prompted captions.

## Timeloop movies
We provide pre-extracted features for CDP on the timeloop movie benchmark here:

[tlm_10movies_cdp_feats](https://drive.google.com/file/d/1-gh7-rb_2bOqxxpJY6dzALO88R9Uf2PP/view?usp=sharing)

And the pre-extracted features for the baseline here:

[tlm_10movies_baseline_feats.pth](https://drive.google.com/file/d/1-yj0qJsUtfS46ghEI76b9sZszXm5D2qP/view?usp=sharing)

And a pre-trained CDPNet model here:

[tlm-arch10_CSPairClsP_p10_e30_model.pkl]()

To run the VideoLlama +CDP at T=+4s, run the following command:

```python
python main_run_unique_captioner.py --dataset tlm --d_model 768 --root storage/grouped_timeloop_feature_seqlen2_10movies.pth --temporal_offset 2  --pp_resume storage/tlm-arch10_CSPairClsP_p10_e30_model.pkl  --pp_threshold -2.0 --max_offset 4 --n_sets 10 --no_cap_default best_margin --enforce_amax 0 --priority none --wbr tlm-cdp-4
```

For the equivalent non-CDP baseline, run the following:

```python
python main_run_unique_captioner.py --dataset tlm --d_model 768 --root storage/grouped_timeloop_feature_seqlen2_numseq6_gap0_start0.pth --temporal_offset 2 --pp_resume tlm-arch10_CSPairClsP_p10_e30_model.pkl  --pp_threshold 2.0 --max_offset 4 --n_sets 12 --no_cap_default lav --comb_maxp 1 --enforce_amax 0 --priority none --wbr tlm-base-4
``` 

If you want to extract your own features, then you can download the frames here:

[timeloop_frames.zip](https://drive.google.com/file/d/1-_rG2azr81cM2GgjtGBTxG62qPSWQr3g/view?usp=sharing)

Our repetition timestamps are here:
[timeloop_timestamps.json]()

## Citation

If you find this work interesting or useful, please cite:

```bibtex
@inproceedings{perrett2024unique,
  title={It's Just Another Day: Unique Video Captioning by Discriminative Prompting},
  author={Perrett, Toby and Han, Tengda and Damen, Dima and Zisserman, Andrew},
  booktitle={ACCV},
  year={2024}
}
```

Our codebase is forked from LaViLa and the benchmark uses Ego4D footage and narrations, so please cite them as well:

```bibtex
@inproceedings{zhao2023learning,
  title={Learning Video Representations from Large Language Models},
  author={Zhao, Yue and Misra, Ishan and Kr{\"a}henb{\"u}hl, Philipp and Girdhar, Rohit},
  booktitle={CVPR},
  year={2023}
}
```
```bibtex
@inproceedings{grauman2022ego4d,
  title={Ego4d: Around the world in 3,000 hours of egocentric video},
  author={Grauman, Kristen and Westbury, Andrew and Byrne, Eugene and Chavis, Zachary and Furnari, Antonino and Girdhar, Rohit and Hamburger, Jackson and Jiang, Hao and Liu, Miao and Liu, Xingyu and others},
  booktitle={CVPR},
  year={2022}
}
```

