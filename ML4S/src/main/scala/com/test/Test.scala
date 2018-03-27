package com.test

import com.ml4s.np.{ NP => np }
import com.ml4s.np.NPImplicits._
import org.nd4s.Implicits._
import org.nd4j.linalg.ops.transforms.Transforms
import com.ml4s.sklearn.activations.Relu
import com.ml4s.sklearn.activations.Softmax

object Test {  
  
  def main(args:Array[String]):Unit={
    val arr=np.array(Array(1,2,3,4),shape=Array(4,1))
    val exponents = np.exp(arr)
    val denominator = np.sum(exponents)
    val softmax = np.divide(exponents, denominator)
    println("arr ",arr)
    println("exponents ",exponents)
    println("denominator ",denominator)
    println("softmax ",softmax)
  }
  
  def npFun()={
    val arr=np.array(Array(1,2,3,4,5,6),shape=Array(2,3))
    val arr2=np.array(Array(1,2,3,4,5,6),shape=Array(3,2))
    
    val subarray = arr.subArray(Array(0,0), Array(2,2), Array(2,2))
    println(subarray)
  }
  
}