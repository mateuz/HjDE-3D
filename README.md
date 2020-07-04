# HjDE-3D: GPU-based Hybrid jDE to the 3D-AB PSP

#### If you use my approach or any adaptation of it, please refer to the following papers:

Boiani M., Parpinelli R. S. **A GPU-based hybrid jDE algorithm applied to the 3D-AB protein structure prediction**. Swarm and Evolutionary Computation (2020): 100711. doi: 10.1016/j.swevo.2020.100711, [URL](https://www.sciencedirect.com/science/article/abs/pii/S2210650220303643)

Boiani M., Dominico G., Stubs Parpinelli R. (2020) **A GPU-Based jDE Algorithm Applied to Continuous Unconstrained Optimization**. In: Abraham A., Cherukuri A., Melin P., Gandhi N. (eds) Intelligent Systems Design and Applications. ISDA 2018. Advances in Intelligent Systems and Computing, vol 940. Springer, Cham. doi: 10.1007/978-3-030-16657-1_85, [URL](https://link.springer.com/chapter/10.1007/978-3-030-16657-1_85)

***

###### Description: Protein Structure Prediction (PSP) problem is an open problem in bioinformatics and, as the problem scales, complexity and processing time increases. In this way, robust methods and massively parallel architectures are required. This repository provide a GPU-based hybrid algorithm, named cuHjDE, to handle the 3D-AB off-lattice PSP problem. The cuHjDE is composed of the jDE algorithm and the Hooke-Jeeves local search algorithm. An important feature present in the proposed method is the use of a crowding mechanism to avoid premature convergence promoting diversification in the search space.


***
##### Requirements

- ##### [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) (tested with 9.2)

- ##### GPU Compute Capability (tested with versions 5.2, 6.1, and 7.0)

- ##### [Boost C++ Libraries - Program Options](https://www.boost.org/) (tested with 1.58.0)

##### Architecture

The Figure below depicts a flow chart of the GPU-based implementation.  In kernels’ boxes, the labels S1, S2, and S3 represent the kernel structure and letters R,SM, and C denote the use of random number generator, shared memory and constant memory, respectively
 
<img src="https://raw.githubusercontent.com/mateuz/HjDE-3D/master/assets/hjde3d-architecture.png" height="800" width="650">

##### Compile

```sh
$ cd repo
$ make
```

##### Parameters Setting

```
$ "runs, r"      - Number of Executions
$ "pop_size, p"  - Population Size
$ "dim, d"       - Protein Length {13, 21, 34, 38, 55, 64, 98, 120}
$ "max_eval, e"  - Number of Function Evaluations
$ "help, h"      - Show this help
```

##### Proteins Added

| PDB ID | PSL |  D  | AB Sequence                                                                                        |
|:------:|:---:|:---:|----------------------------------------------------------------------------------------------------|
|  1BXP  |  13 |  21 | ABBBBBBABBBAB                                                                                      |
|  1CB3  |  13 |  21 | BABBBAABBAAAB                                                                                      |
|  1BXL  |  16 |  27 | ABAABBAAAAABBABB                                                                                   |
|  1EDP  |  17 |  29 | ABABBAABBBAABBABA                                                                                  |
|  2ZNF  |  18 |  31 | ABABBAABBABAABBABA                                                                                 |
|  1EDN  |  21 |  37 | ABABBAABBBAABBABABAAB                                                                              |
|  2H3S  |  25 |  45 | AABBAABBBBBABBBABAABBBBBB                                                                          |
|  1ARE  |  29 |  53 | BBBAABAABBABABBBAABBBBBBBBBBB                                                                      |
|  2KGU  |  34 |  63 | ABAABBAABABBABAABAABABABABABAAABBB                                                                 |
|  1TZ4  |  37 |  69 | BABBABBAABBAAABBAABBAABABBBABAABBBBBB                                                              |
|  1TZ5  |  37 |  69 | AAABAABAABBABABBAABBBBAABBBABAABBABBB                                                              |
|  1AGT  |  38 |  71 | AAAABABABABABAABAABBAAABBABAABBBABABAB                                                             |
|  1CRN  |  46 |  87 | BBAAABAAABBBBBAABAAABABAAAABBBAAAAAAAABAAABBAB                                                     |
|  2KAP  |  60 | 115 | BBAABBABABABABBABABBBBABAABABAABBBBBBABBBAABAAABBABBABBAAAAB                                       |
|  1HVV  |  75 | 145 | BAABBABBBBBBAABABBBABBABBABABAAAAABBBABAABBABBBABBAABBABBAABBBBBAABBBBBABBB                        |
|  1GK4  |  84 | 163 | ABABAABABBBBABBBABBABBBBAABAABBBBBAABABBBABBABBBAABBABBBBBAABABAAABABAABBBBAABABBBBA               |
|  1PCH  |  88 | 171 | ABBBAAABBBAAABABAABAAABBABBBBBBABAAABBBBABABBAABAAAAAABBABBABABABABBABBAABAABBBAABBAAABA           |
|  2EWH  |  98 | 191 | AABABAAAAAAABBBAAAAAABAABAABBAABABAAABBBAAAABABAAABABBAAABAAABAAABAABBAABAAAAABAAABABBBABBAAABAABA |

##### Execute

```sh
$ cd repo
$ ./demo <parameter setting> or make run (with default parameters)
```

##### Clean up

```sh
$ make clean
```

##### TODO

- Empty list for a while

***

[1] J. Brest, V. Zumer and M. S. Maucec, "Self-Adaptive Differential Evolution Algorithm in Constrained Real-Parameter Optimization," 2006 IEEE International Conference on Evolutionary Computation, Vancouver, BC, 2006, pp. 215-222. doi: 10.1109/CEC.2006.1688311, [URL](http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1688311&isnumber=35623)

[2] [CUDA is a parallel computing platform and programming model developed by NVIDIA for GPGPU](https://developer.nvidia.com/cuda-zone)
