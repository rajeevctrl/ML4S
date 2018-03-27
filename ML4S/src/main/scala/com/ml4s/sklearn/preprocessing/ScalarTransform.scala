package com.ml4s.sklearn.preprocessing

import org.nd4j.linalg.api.ndarray.INDArray
import com.ml4s.np.{NP => np}
import com.ml4s.np.NP._
import org.nd4s.Implicits._

class ScalarTransform(withMean:Boolean=true,withStd:Boolean=true) {
  var colMean:INDArray = _
  var colStd:INDArray = _
  
  /**
   * Fits the transforms but does not modify data.
   */
  def fit(X:INDArray)={
    this.colMean = X.mean(0)
    this.colStd = X.std(0)
  }
  
  /**
   * Fits the transforms with {@code X} as training data. Then applies the transformation on X and returns transformed values.
   */
  def fitTransform(X:INDArray):INDArray={
    this.fit(X)
    val trans = this.transform(X)
    return trans
  }
  
  def transform(X:INDArray):INDArray={
    val trans = (X - this.colMean) / this.colStd
    return trans
  }
}

object ScalarTransform{
  def apply(withMean:Boolean=true,withStd:Boolean=true)={
    new ScalarTransform(withMean,withStd)
  }
}