package com.ml4s.sklearn.activations

import org.nd4j.linalg.api.ndarray.INDArray
import com.ml4s.np.NPImplicits._
import org.nd4s.Implicits._

class Tanh extends Activation {
  
   /**
   * Activation function.
   */
  def activation(arr1:INDArray):INDArray={
    var arr=arr1
    val posExp = np.exp(arr)
    val negExp = np.exp(-arr)
    val tanh = (posExp - negExp)/(posExp + negExp)
    return tanh
  }
  
  /**
   * Derivative of activation function.
   */
  def derivative(arr:INDArray):INDArray={
    val g_z =activation(arr)
    val derivative = 1 - (g_z * g_z)
    return derivative
  }
  
}