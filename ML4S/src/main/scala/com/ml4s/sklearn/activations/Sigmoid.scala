package com.ml4s.sklearn.activations

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.ops.transforms.Transforms
import com.ml4s.np.{NP => np}
//import com.ml4s.np.NP._
import com.ml4s.np.NPImplicits._
import org.nd4s.Implicits._



class Sigmoid extends Activation {
  
  /**
   * Activation function.
   */
  def activation(arr1:INDArray):INDArray = {
    var arr=arr1
    val sig = 1 / (1 + np.exp(-1*arr))
    return sig
  }
  /**
   * Derivative of activation function.
   */
  def derivative(arr:INDArray):INDArray={
    val sig=activation(arr)
    val derivative = sig * (1 - arr)
    return derivative
  }

  
}