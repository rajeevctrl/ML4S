package com.ml4s.sklearn.activations


import com.ml4s.np.{NP => np}
import com.ml4s.np.NPImplicits._
import org.nd4s.Implicits._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.BooleanIndexing
import org.nd4j.linalg.indexing.conditions.Conditions

class Relu extends Activation {
  /**
   * Implementation of activation function for given input.
   */
  def activation(a:INDArray):INDArray={
    //Create a zeros array of same shape as that of 'a'.
    val zeros = np.zeros(shape=a.shape())
    //Relu is max of zero or given value.
    val relu = np.max(a, zeros)
    return  relu
  }
  
  /**
   * Implementation of derivative of activation function for given input.
   */
  def derivative(a:INDArray):INDArray={
    var arr = a.dup()
    //Replace values less than and equal to zero with zero.
    BooleanIndexing.replaceWhere(arr, 0.0, Conditions.lessThanOrEqual(0.0))
    //Replace values more than zero with 1.
    BooleanIndexing.replaceWhere(arr, 1.0, Conditions.greaterThan(0.0))
    return arr
  }
  
  
}