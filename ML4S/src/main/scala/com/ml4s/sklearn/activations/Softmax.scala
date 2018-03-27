package com.ml4s.sklearn.activations

import org.nd4j.linalg.api.ndarray.INDArray
import com.ml4s.np.{NP => np}
import com.ml4s.np.NPImplicits._
import org.nd4s.Implicits._

class Softmax extends Activation {
  def activation(a:INDArray):INDArray={
    val exponents = np.exp(a)
    val denominator = np.sum(exponents)
    val softmax = np.divide(exponents, denominator)
    return softmax
  }
  
  
  def derivative(a:INDArray):INDArray={
    return null
  }
}