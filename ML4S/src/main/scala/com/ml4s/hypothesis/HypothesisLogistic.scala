package com.ml4s.hypothesis

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

class HypothesisLogistic extends Hypothesis {
  
  override def hypothesis(X:INDArray, theta:INDArray):INDArray={
    val z=X.mmul(theta)
    val sigmoid=org.nd4j.linalg.ops.transforms.Transforms.sigmoid(z)
    return sigmoid
  }
  
  override def hypothesisFirstDerivative(X:INDArray, theta:INDArray):INDArray={
    val featureCount=X.size(1)
    val z=X.mmul(theta)
    val sigmoid=org.nd4j.linalg.ops.transforms.Transforms.sigmoid(z)
    val ones=Nd4j.ones(X.size(0)).transpose()
    var derivativeSigmoid=(ones.sub(sigmoid)).mul(sigmoid)
    //Each col of X is a dimension and derivative is computed w.r.t each dimension.
    for(i <- Range(0,featureCount-1)){
      derivativeSigmoid=Nd4j.hstack(derivativeSigmoid,derivativeSigmoid)
    }
    //Derivative is to be computed for each dimension so multiply derivative of sigmoid with each dimension.
    val derivativeHypo=derivativeSigmoid.mul(X)
    return derivativeHypo
  }
}

object HypothesisLogistic{
  
  def main(args:Array[String]):Unit={
    /*val nd = Nd4j.create(Array[Float](1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), Array[Int](2, 6));
    val ndv = org.nd4j.linalg.ops.transforms.Transforms.sigmoid(nd);
    println(ndv)*/
    val x=Nd4j.create(Array[Double](1,2,3,4),Array[Int](4,1))
    val ones=Nd4j.ones(4).transpose()
    val derivative=(ones.sub(x)).mul(x)
    println(derivative)
  }
}