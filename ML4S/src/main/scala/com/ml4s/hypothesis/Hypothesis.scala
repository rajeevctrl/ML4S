package com.ml4s.hypothesis

import org.nd4j.linalg.api.ndarray.INDArray


class Hypothesis{
  def hypothesis(X:INDArray, theta:INDArray):INDArray={return null}
  def hypothesisFirstDerivative(X:INDArray, theta:INDArray):INDArray={return null}
}

/*object Hypothesis extends Hypothesis{
  type hypothesisTemplate = (INDArray, INDArray) => INDArray
  type hypothesisFirstDerivativeTemplate = (INDArray, INDArray) => INDArray
  
  
}*/


