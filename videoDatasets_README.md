# Datasets

This README acts as the training & testing guideline for all our experiments.

## Saving

All SPAD emulations are saved as 3 channel images within [0,1] at respective dataset's native resolution. 
(To visualize, multiply by 255) 

## 0. japan-alley: A synthetic scene bought from an artist (used for early day sanity & baseline tests)

### japan-alley-static - 1 video

Little to no-motion testing: lamps swinging and tree branches swaying.
DR testing: the lamps turn on for dynamic range testing.
Non-lambertian testing: Puddle on the ground tests non-lambertian surfaces somewhat.

### japan-alley-dynamic - 1 video

Fast motion panning across the alley + lamps swaying motion 

## 1. visionsim - Purely synthetic blender scenes - 50 videos

Natively rendered at 100 fps. 50 color scenes at 800x800 resolution.
The dataset also comes with flows, depth, segmentation, normals and camera poses.

> We utilize this dataset twice at 2 different fps.

### Mode A - Native fps for Perfect Flow Training of Video Model
We use the ground truth flow without interpolating these scenes to not introduce interpolation errors + more motion at lesser fps to build an understanding of motion early. Later stages of video model training are to rectify temporal issues (flickering etc.) and potentially understand extremely high-speed phenomena (explosions, non-rigid/fluid motion)

### Mode B - Interpolated 4x for Second Stage Video Model Progression.
Since the next available fps we have is 1000fps, making a jump from 100 to 1000 would be challenging for the model to handle. So we interpolate visionsim 4x (or to 400 fps) for second fps-progression of our video model training.


## 2. I2-2000fps - 2000fps color, 512 x 1024, 280 scenes - real captures

Continues fps progression


## 3. XVFI - 1000fps color, 4K, 140 scenes - real captures. - 4438 clips
4096×2160 cropped to 768x768. Each video with 5,000 frames (5 seconds)
Natural transition after 400 fps, no need to interpolate further.


--- Train The Video Model + RAFT-loss with the aforementioned^^^^^^^ (Deformable needs to be a separate training cycle)

### Dataset Statistics so far:

Total frames: 1,125,318
Videos: 4818

Training: 1125318 - 7317 = 1118001 
Testing/Validation: 990 (xvfi) + 6327 (i22k)= 7317

---

## 4. xd-scraped: Extreme Deformable Videos Scraped from the Internet (322+68=390 videos)

Used for last stage fine-tuning of the video model. Wild card fps. High resolution.

<!-- TODO: Find N -->
Each video is atleast 10 seconds/ > 200-frames.

Very interesting deformable physics scenes - explosions, shattering objects, bullets, fluids etc.

> All the videos have been manually filtered for interesting content (subjective binary decision) and then processed: removed audio --> clipped --> resized --> spadified 

Credits: 
1. www.youtube.com/@theslowmoguys 
2. www.youtube.com/@BallisticHighSpeed

---

## 5. UDM10 - 10 videos at 24fps

---

## 6. SPMC Videos - 30 videos at 24fps

---

## 7. REDS - 240 videos at 120fps

---

TODO:
### Dataset Statistics so far (Not including xd-scraped):

Total frames: 1,424,188 -> 2.7M
Videos: 5098 -> 42k

Training: 1,424,188 - 7317
Testing/Validation: 990 (xvfi) + 6327 (i22k) = 7317

---

TODO:
## Video Model Training Progression

| Stage |            Dataset           |  FPS      | Purpose |
|-----  |----------------------------  |:----:     |-------  |
| 1     | UDM10 + SPMC + YouHQ         | 24        | Just like WarmupLR -- but build most of video prior + temp consistency here  | 
| 2     | visionsim-untouched (Mode A) | 100       | Foundational motion + clean GT flow                                  |
| 3     | REDS 120fps                  | 120       | Approximated flow but real scenes - reinstall realism!               |
| 4     | visionsim-4x (Mode B)        | 400       | Intermediate transition fps                                          |
| 5     | XVFI                         | 1000      | Real captures, temporal structure                                    |
| 6     | I2-2000fps                   | 2000      | Real high-fps captures                                               |
| 7     | xd-scraped                   | 3k - 200k | Non-rigid, fluid, deformable physics | 390 scenes * 200+ frames |
| 8     | Mixed fps training           | 24 - 200k | Boost varying fps producing capabilities                        |


## Real SPAD capture datasets 
#### A - wacv_vision: "Burst Photography using Single Photon Cameras", Ma et al. 2023. All sequences captured at 10kHz.
#### B - qbp: Quanta Burst Photography, Ma et al. 2020. Most sequences captured at 10--20kHz

## VLMs for captioning

1.  InternVL3-8B ranked 25th on the opencompass vlm leaderboard (above Gemini 2.0 Pro, Claude 3.5 Sonnet-2024-10-22 and GPT-4o (0806, detail-high version))
2. Llava-onevision-qwen2-7b-si ranked 120 (Above all Llava-Next (162) and Llava (247) variants)
3. Using no prompts (null)

opencompass open-vlm-leaderboard:

@inproceedings{duan2024vlmevalkit,
  title={Vlmevalkit: An open-source toolkit for evaluating large multi-modality models},
  author={Duan, Haodong and Yang, Junming and Qiao, Yuxuan and Fang, Xinyu and Chen, Lin and Liu, Yuan and Dong, Xiaoyi and Zang, Yuhang and Zhang, Pan and Wang, Jiaqi and others},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  pages={11198--11201},
  year={2024}
}

