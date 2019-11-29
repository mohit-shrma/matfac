## Code for matrix completion
This repository contains scalable matrix factorization code using SGD, ALS and CCD++. Also, it includes code for the paper "Adaptive Matrix Completion for the Users and the Items in Tail" by Mohit Sharma and George Karypis, University of Minnesota (The Web Conference'19, formerly WWW). The researchers or users should look for following source files for implementation details of methods in the paper:

* Matrix Factorization (MF) - modelMF.h and modelMF.cpp
* IFWMF - modelInvPopMF.h and modelInvPopMF.cpp
* TMF - modelDropoutSigmoid.h and modelDropoutSigmoid.cpp
* TMF + Dropout - modelPoissonDropout.cpp and modelPoissonDropout.h


## Citing:
If you refer or use any part of this code, please cite it using the following BibTex entry:
```
@inproceedings{sharma2019adaptive,
  title={Adaptive matrix completion for the users and the items in tail},
  author={Sharma, Mohit and Karypis, George},
  booktitle={The World Wide Web Conference},
  pages={3223--3229},
  year={2019},
  organization={ACM}
}```
