## A software implementation of the Monte Carlo method for propagation of distributions.

The input quantities described by TSP or normal distributions. Possible options for specifying TSP:

* by parameters:
```
"v": {"type": "tsp", "a": 0.1, "m": 0.3, "b": 0.7, "p": 1.5},
``` 
* by a mathematical expectation, uncertainty and coverage interval (or coverage factor, or boundaries):
```
"x1": {"type": "tsp", "mu": 0.6, "u": 0.164, "c1": 0.23, "c2": 0.87, "cov_prob": 0.95}
"x2": {"type": "tsp", "mu": 0.5, "u": 0.164, "cov_factor": 1.99, "cov_prob": 0.95}
"x3": {"type": "tsp", "a": 0.0, "b": 1.0, "mu": 0.6, "u": 0.16}
```
* by experimental data (histogram or sample):
```
"z1": {"type": "tsp", "sample_path": "./test_sample.txt"}
"z2": {"type": "tsp", "histogram_path": "./test_hist.txt"}
```

In case of normal distribution:
```
"t": {"type": "normal", "mu": 1.0, "u": 0.01}
```

Multiple inputs could be describet at once: `"v1, v2, v3": { ... }`

<br>

Usage:
```
python mc.py path/to/task.json
```
For an example of the task description please see [an example](./test_task.json)

The output will be saved in the same folder
