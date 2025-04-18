# Mini-TensorFlow
## Overview
Mini-TensorFlow (mini-tf) is a little project of mine that aims to build a TensorFlow-v1 like deep learning framework.

Its current features include:

- Declarative style computational graph definition 
- A complete set of useful operators, with forward and backward methods:
	- +, -, *, /, <, >, ==, power and logarithm
	- Mean, LaynerNorm, and Broadcasting
	- ReLU and Softmax activation
- Decoupled computational graph and computation; computation is done by feeding in numeric arrays to the evaluator  
- Support for operator fusion and overloading
- Automatic backward computational graph construction with one method call
- Accelerated computation with CUDA or Metal Performance Shader (MPS)


## Usage
The computational backbone of this framework is (ironically) PyTorch. This is because this project is a proof-of-concept to recreate TensorFlow, with its focus being building the framework and implementating the class and methods to support decoupled graph definition and computation.

To test it for yourself, type in terminal
```bash
pip install -e path_to_cloned_repo
``` 
After installing the package, import it:
```python
import mini_tf.auto_diff as ad
import numpy

x1 = ad.Variable(name="x1")
x2 = ad.Variable(name="x2")

x1_value = np.array(2)
x2_value = np.array(3)
y_value = evaluator.run(input_dict={x1: x1_value, x2: x2_value})
...
```
For more, checkout `tests` and `demo` folder


## Test
To demonstrate the flexiness of this project, I have added a demo, in which I build a transformer block with a MLP classifier head, to run classification on the MNIST dataset. It runs with stochastic gradient descent algorithm and softmax loss. \
There are also a few tests in the `tests` folder. Run them with
```bash
pytest ./tests/name_of_test_file.py
```
For fused operators performance testing specifically, run the script with `python` instead of `pytest`.


## Credits
Credits to UCSD CSE234 staff for starter code
