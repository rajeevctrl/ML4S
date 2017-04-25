package com.test

import org.nd4j.linalg.factory.Nd4j
import com.ml4s.driver.LinearRegressionDriver
import com.ml4s.driver.LogisticRegressionDriver
import org.nd4j.linalg.util.MathUtils

object Test {
  
  def main(args:Array[String]):Unit={
    /*for(i <- Range(0,10)){
      val num=MathUtils.randomDoubleBetween(0.0, 0.1)
      println(num)
    }*/
    
    test()
  }
  
  def test()={
    //val arr=Array(1.0,10.0,2.0,20.0,3.0,30.0,4.0,40.0);
    val arr=Array[Double](1,2,3,4,5,6,7,8,9)
    /*val arr=Array[Array[Double]](Array[Double](0,0),Array[Double](0,1),Array[Double](1,0),Array[Double](1,1),
        Array[Double](3,3),Array[Double](3,4),Array[Double](4,3),Array[Double](4,4))*/
    val X=Nd4j.create(arr,Array(9,1))
    //val arrY=Array(0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0)
    val arrY=Array[Double](1,2,3,4,5,6,7,8,9)
    val Y=Nd4j.create(arrY,Array(9,1))
    /*val arrThetas=Array(0.2)
    val thetas=Nd4j.create(arrThetas,Array(2,1))*/
    val alpha=0.000001
    val iter=100000000
    val driver=new LinearRegressionDriver(X,Y,alpha,iter);
    driver.fitModel()
    println(driver.getTheta())
    driver.saveModel("/home/rajeev/Documents/temp/Model folder/b.model")
    driver.setTheta(null)
    println(driver.getTheta())
    driver.loadModel("/home/rajeev/Documents/temp/Model folder/b.model")
    println(driver.getTheta()+"   "+driver.getTrainingError())
    val hypo=driver.getHypothesis()
    val prod=hypo.hypothesis(X, driver.getTheta())
    println(prod)
    
    val predictedValues=driver.predict(X)
    println("Predicted values")
    println(predictedValues)
  }
}