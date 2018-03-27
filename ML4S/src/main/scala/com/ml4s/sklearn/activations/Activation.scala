package com.ml4s.sklearn.activations

import org.nd4j.linalg.api.ndarray.INDArray

trait Activation {
  
  def activation(a:INDArray):INDArray
  def derivative(a:INDArray):INDArray
}