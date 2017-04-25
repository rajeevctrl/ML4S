package com.ml4s.hypothesis

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import java.io.Serializable

class HypothesisLinear extends Hypothesis{
  
  /**
   * This function creates a linear hypothesis for given input values.
   * 
   * @param X: Feature matrix of training data. Dimensions of matrix are NUM_TRAINING_EXAMPLES x DIMENSIONS
   * @param Y: Output vector of training data. This is a column vector with dimensions NUM_TRAINING_EXAMPLES x 1
   */
  
  override def hypothesis(X:INDArray,thetas:INDArray):INDArray={
    //In linear hypothesis each X row value is multiplied with thetas to compute a matrix/vector equal in dimension to that of Y.
    val hypothesis=X.mmul(thetas);
    return hypothesis;
  }
  
  /**
   * For linear hypothesis first derivative w.r.t thetas is X itself.
   * 
   * @param X: Feature matrix of training data. Dimensions of matrix are NUM_TRAINING_EXAMPLES x DIMENSIONS
   * @param Y: Output vector of training data. This is a column vector with dimensions NUM_TRAINING_EXAMPLES x 1
   */
  override def hypothesisFirstDerivative(X:INDArray, thetas:INDArray):INDArray={
    return X;
  }
  
  
  /*def main(args:Array[String]):Unit={
    val arr=Array(1.0,10.0,2.0,20.0,3.0,30.0,4.0,40.0);
    val X=Nd4j.create(arr,Array(4,2))
    val arrY=Array(11.0,12.0,13.0,14.0)
    val Y=Nd4j.create(arrY,Array(4,1))
    val arrThetas=Array(0.2,0.2)
    val thetas=Nd4j.create(arrThetas,Array(2,1))
    val result=linearHypothesis(X, thetas)
    println(result)
  }*/
}