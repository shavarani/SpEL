<p align="center" width="100%"><img src="resources/logo.png" width="30%" alt="SpEL"></p>

---

**SpEL** (**S**tructured **p**rediction for **E**ntity **L**inking)  is a structured prediction entity linking approach 
that uses new training and inference ideas obtaining a new state of the art on Wikipedia entity linking, with better 
compute efficiency and faster inference than previous methods. 
It was proposed in our paper [SpEL: Structured Prediction for Entity Linking](https://arxiv.org/abs/2306.00000).
It outperforms the state of the art on the commonly used 
[AIDA benchmark](https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/research/ambiverse-nlu/aida/downloads) 
dataset for entity linking to Wikipedia. Apart from being more accurate, it also is the most compute efficient in terms 
of number of parameters and speed of inference.

The following figure schematically depicts SpEL:

<p align="center" width="100%"><img src="resources/SpEL.png" width="60%" alt="SpEL"></p>

This repository contains the source code to finetune RoBERTa models and evaluate them using [GERBIL](https://github.com/dice-group/gerbil).

---

Entity Linking evaluation results of *SpEL* compared to that of the literature over AIDA test sets:

| Approach                                                  | EL Micro-F1<br/>test-a |   EL Micro-F1<br/>test-b    |           #params<br/>on GPU            | speed<br/>sec/doc  |
|-----------------------------------------------------------|:----------------------:|:---------------------------:|:---------------------------------------:|:------------------:|
| Hoffart et al. (2011)                                     |          72.4          |            72.8             |                    -                    |         -          |
| Kolitsas et al. (2018)                                    |          89.4          |            82.4             |                 330.7M                  |       0.097        |
| Broscheit (2019)                                          |          86.0          |            79.3             |                 495.1M                  |       0.613        |
| Peters et al. (2019)                                      |          82.1          |            73.1             |                    -                    |         -          |
| Martins et al. (2019)                                     |          85.2          |            81.9             |                    -                    |         -          |
| van Hulst et al. (2020)                                   |          83.3          |            82.4             |                  19.0M                  |       0.337        |
| Févry et al. (2020)                                       |          79.7          |            76.7             |                    -                    |         -          |
| Poerner et al. (2020)                                     |          90.8          |            85.0             |                 131.1M                  |         -          |
| Kannan Ravi et al. (2021)                                 |           -            |            83.1             |                    -                    |         -          |
| De Cao et al. (2021b)                                     |           -            |            83.7             |                 406.3M                  |       40.969       |
| De Cao et al. (2021a) (no mention-specific candidate set) |          61.9          |            49.4             |                 124.8M                  |       0.268        |
| De Cao et al. (2021a) (using PPRforNED candidate set)     |          90.1          |            85.5             |                 124.8M                  |       0.194        |
| Mrini et al. (2022)                                       |           -            |            85.7             |     (train) 811.5M / (test) 406.2M      |         -          |
| Zhang et al. (2022)                                       |           -            |            85.8             |                 1004.3M                 |         -          |
| Feng et al. (2022)                                        |           -            |            86.3             |                 157.3M                  |         -          |
| <hr/>                                                     |         <hr/>          |            <hr/>            |                  <hr/>                  |       <hr/>        |
| **SpEL** (no mention-specific candidate set)              |          90.9          |            84.2             |                 128.9M                  |       0.094        |
| **SpEL** (KB+Yago candidate set)                          |          90.1          |            84.7             |                 128.9M                  |       0.161        |
| **SpEL** (PPRforNED candidate set) (context-agnostic)     |          91.5          |            86.1             |                 128.9M                  |       0.158        |
| **SpEL** (PPRforNED candidate set) (context-aware)        |          92.4          |            87.5             |                 128.9M                  |       0.157        |
